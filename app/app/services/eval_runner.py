# -*- coding: utf-8 -*-
# Objective: Execute the prompts of one stored evaluation run through the routing stack.
"""Per-prompt execution of an evaluation run, separated from the Celery task.

``tasks.task_execute_eval_run`` only handles status, the frozen policy and
retries; this module builds each ``QueryRequest`` from the prompt catalog,
runs it, records the result (or the error) and accumulates the metrics that
make up the run summary.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException

from ..benchmark_catalog import resolve_catalog_asset_path
from ..schemas import QueryRequest, WorkloadHints

AddResult = Callable[..., Any]
ProcessQuery = Callable[[QueryRequest], Awaitable[Dict[str, Any]]]


@dataclass(frozen=True)
class EvalRunOptions:
    modality: str = "text"
    use_cache: bool = False
    max_tokens: int = 512
    temperature: float = 0.5


@dataclass
class EvalTotals:
    quality: List[float] = field(default_factory=list)
    latency: List[float] = field(default_factory=list)
    cost: List[float] = field(default_factory=list)

    def add(self, quality: float, latency_s: float, cost_usd: float) -> None:
        self.quality.append(quality)
        self.latency.append(latency_s)
        self.cost.append(cost_usd)


def catalog_index(run_metadata: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``query -> catalog item`` for the prompts that came from a benchmark catalog."""
    catalog = run_metadata.get("prompt_catalog") or []
    return {str(item.get("query", "")).strip(): item for item in catalog if isinstance(item, dict)}


def frozen_settings(run_metadata: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Whether the run executes under a frozen policy, and the config snapshot to freeze."""
    manifest = run_metadata.get("experiment_manifest") or {}
    frozen = bool(run_metadata.get("frozen_policy")) or bool(manifest.get("frozen_policy"))
    return frozen, manifest.get("config_snapshot")


def _workload_hints(item: Dict[str, Any]) -> Optional[WorkloadHints]:
    theme, benchmark_id = item.get("theme"), item.get("id")
    if not (theme or benchmark_id):
        return None
    return WorkloadHints(
        theme=str(theme) if theme else None,
        benchmark_id=str(benchmark_id) if benchmark_id else None,
    )


def build_eval_request(
    prompt_text: str, item: Dict[str, Any], run: Dict[str, Any], opts: EvalRunOptions
) -> QueryRequest:
    """QueryRequest for one prompt; a catalog ``image_path`` turns it into a multimodal query."""
    image_b64: Optional[str] = None
    modality = opts.modality
    if item.get("image_path"):
        image_b64 = base64.b64encode(resolve_catalog_asset_path(str(item["image_path"])).read_bytes()).decode("ascii")
        modality = "multimodal"
    return QueryRequest(
        query=prompt_text,
        modality=modality,
        images=[image_b64] if image_b64 else None,
        image_b64=image_b64,
        enable_rag_for_answer=bool(item.get("enable_rag")),
        use_cache=opts.use_cache,
        max_tokens=opts.max_tokens,
        temperature=opts.temperature,
        policy_version=run.get("policy_version"),
        tenant_id=run.get("tenant_id"),
        workload_hints=_workload_hints(item),
    )


def result_metadata(run: Dict[str, Any], resp: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    meta = resp.get("metadata", {})
    return {
        "policy_version": run.get("policy_version"),
        "answer": str(resp.get("answer") or ""),
        "grounded": bool(meta.get("grounded")),
        "verification_status": meta.get("verification_status"),
        "abstained": bool(meta.get("abstained")),
        "knowledge_version": meta.get("knowledge_version"),
        "confidence_score": meta.get("confidence_score"),
        "workload_class": meta.get("workload_class"),
        "detected_complexity": meta.get("detected_complexity"),
        "benchmark_theme": item.get("theme"),
        "benchmark_id": item.get("id"),
        "reference": item.get("reference"),
        "attack_strategy": item.get("attack_strategy"),
    }


def response_metrics(resp: Dict[str, Any]) -> Tuple[float, float, float]:
    """Quality, latency (s) and cost (USD) reported by one routed answer."""
    quality = float(resp.get("metadata", {}).get("quality", 0.0) or 0.0)
    latency_s = float(resp.get("latency_s", 0.0) or 0.0)
    cost = float(resp.get("estimated_cost_usd", resp.get("cost_per_1k", 0.0)) or 0.0)
    return quality, latency_s, cost


def error_metadata(exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, HTTPException):
        return {"error": str(exc.detail), "status_code": exc.status_code}
    return {"error": str(exc)}


async def run_eval_prompts(
    run_id: str,
    run: Dict[str, Any],
    opts: EvalRunOptions,
    add_result: AddResult,
    process_query: Optional[ProcessQuery] = None,
) -> EvalTotals:
    """Route every prompt of the run; a failing prompt is recorded as ``model="error"`` and skipped."""
    if process_query is None:
        from .query_runtime import process_query_request

        process_query = process_query_request
    by_query = catalog_index(run.get("metadata") or {})
    totals = EvalTotals()
    for prompt in run.get("prompts") or []:
        prompt_text = str(prompt)
        item = by_query.get(prompt_text.strip(), {})
        try:
            wrapped = await process_query(build_eval_request(prompt_text, item, run, opts))
            resp = wrapped.get("result", wrapped)
            quality, latency_s, cost = response_metrics(resp)
            add_result(
                run_id=run_id,
                prompt_text=prompt_text,
                model=str(resp.get("model") or "unknown"),
                quality=quality,
                latency_s=latency_s,
                cost_usd=cost,
                metadata=result_metadata(run, resp, item),
            )
            totals.add(quality, latency_s, cost)
        except Exception as exc:
            add_result(
                run_id=run_id,
                prompt_text=prompt_text,
                model="error",
                quality=0.0,
                latency_s=0.0,
                cost_usd=0.0,
                metadata=error_metadata(exc),
            )
    return totals


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(n_prompts: int, totals: EvalTotals) -> Dict[str, Any]:
    """Run summary: means over the prompts that produced an answer."""
    return {
        "finished_at": time.time(),
        "n": n_prompts,
        "quality_mean": _mean(totals.quality),
        "latency_mean": _mean(totals.latency),
        "cost_mean": _mean(totals.cost),
    }
