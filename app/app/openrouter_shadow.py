# -*- coding: utf-8 -*-
# Objective: Shadow comparison between an explored OpenRouter model and the incumbent.
"""Replay a sample of explored queries on the incumbent model and compare quality.

Extracted from ``app.openrouter_explorer`` (re-exported there).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from app.openrouter_exploration_state import _load_model_stats, _save_model_stats

logger = logging.getLogger("app.openrouter_explorer")


_SHADOW_PREFIXES = ("openrouter/", "openai/", "anthropic/", "gemini/", "ollama/")
_SHADOW_HISTORY = 20


def _shadow_eligible(explored_model: str, incumbent_model: Optional[str]) -> bool:
    if not incumbent_model or incumbent_model == explored_model:
        return False
    return incumbent_model.startswith(_SHADOW_PREFIXES)


def _judged_quality(judge_scores: Any) -> float:
    """Mean judge score on the 0-100 scale used by the explorer (5/10 when no judge answered)."""
    valid = [s["score"] for s in judge_scores if "score" in s]
    return round((float(np.mean(valid)) if valid else 5.0) * 10.0, 2)


def _append_shadow(stats: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the last ``_SHADOW_HISTORY`` paired deltas and their mean quality delta."""
    shadows = stats.get("shadow_comparisons")
    shadows = (shadows if isinstance(shadows, list) else []) + [delta]
    stats["shadow_comparisons"] = shadows[-_SHADOW_HISTORY:]
    stats["shadow_delta_quality_mean"] = round(
        sum(float(s.get("delta_quality", 0)) for s in stats["shadow_comparisons"]) / len(stats["shadow_comparisons"]),
        4,
    )
    return stats


async def maybe_run_shadow_comparison(
    *,
    query: str,
    explored_model: str,
    explored_answer: str,
    explored_quality: float,
    explored_latency: float,
    explored_cost: float,
    incumbent_model: Optional[str],
    deps: Dict[str, Any],
    settings: Any,
) -> None:
    """Run incumbent model on the same query and store paired comparison deltas."""
    if not _shadow_eligible(explored_model, incumbent_model):
        return
    try:
        started = time.perf_counter()
        out, meta = await deps["call_model"](
            model=incumbent_model,
            prompt=query,
            modality="text",
            image_b64=None,
            temperature=float(getattr(settings, "TEMPERATURE_DEFAULT", 0.2) or 0.2),
            max_tokens=int(getattr(settings, "MAX_TOKENS_DEFAULT", 512) or 512),
        )
        incumbent_latency = time.perf_counter() - started
        incumbent_answer = out if isinstance(out, str) else str(out)
        _, _, incumbent_cost, _, _ = deps["parse_meta_cost"](
            meta=meta,
            chosen_model=incumbent_model,
            cost_lookup=deps["get_model_cost"],
        )
        incumbent_quality = _judged_quality(await deps["judge_answer"](query, incumbent_answer))
        delta = {
            "delta_quality": round(explored_quality - incumbent_quality, 3),
            "delta_latency_s": round(explored_latency - incumbent_latency, 3),
            "delta_cost_usd": round(explored_cost - float(incumbent_cost or 0.0), 6),
            "incumbent_model": incumbent_model,
            "explored_model": explored_model,
            "recorded_at": time.time(),
        }

        from app.openrouter_explorer import _get_redis  # import tardio: o explorer reexporta este módulo

        rds = await _get_redis()
        if rds:
            stats = _append_shadow(await _load_model_stats(rds, explored_model), delta)
            await _save_model_stats(rds, explored_model, stats)

        logger.info(
            "[openrouter_explore] shadow %s vs %s delta_q=%.2f delta_cost=%.6f",
            explored_model,
            incumbent_model,
            delta["delta_quality"],
            delta["delta_cost_usd"],
        )
    except Exception as exc:
        logger.warning("[openrouter_explore] shadow comparison failed: %s", exc)
