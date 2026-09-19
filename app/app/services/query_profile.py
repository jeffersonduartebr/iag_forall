# -*- coding: utf-8 -*-
# Objective: Per-request runtime profile (workload class, retrieval mode, deadlines, token budget).
"""Classify a query and derive its runtime profile before routing.

Text hints pick the workload class (simple / reasoning / retrieval) and the
retrieval profile; settings map each class to synchronous deadlines, provider
timeouts and token budgets. Extracted from ``services.query_runtime``
(re-exported there).
"""

from __future__ import annotations

import re
from typing import Any, Dict

from ..observability import QUERY_COMPLEXITY_DETECTED
from ..services.query_complexity import apply_complexity_runtime_adjustments, detect_query_complexity
from ..settings_dynamic import settings

_REASONING_HINTS = (
    "step by step",
    "passo a passo",
    "prove",
    "derive",
    "demonstre",
    "justify",
    "explique detalhadamente",
    "chain of thought",
)
_RETRIEVAL_HINTS = (
    "according to",
    "de acordo com",
    "cite",
    "reference",
    "document",
    "manual",
    "policy",
    "regulation",
    "source",
    "baseado no material",
)
_SIMPLE_QUERY_HINTS = (
    "quanto é",
    "what is",
    "who is",
    "when was",
    "capital of",
    "defina",
    "define",
)
_SOURCE_REQUIRED_HINTS = (
    "fonte",
    "fontes",
    "source",
    "sources",
    "artigo",
    "paper",
    "lei",
    "decreto",
    "resolução",
    "norma",
    "manual",
)



def classify_query_workload(req: Any, modality: str, image_input: str | None) -> str:
    """Classify a request into a small set of performance-relevant workload classes."""
    if image_input or modality in {"vision", "multimodal"}:
        return "vision"

    if str(settings.get("ROUTER_QUERY_CLASSIFIER_ENABLED", "1")).strip() != "1":
        return "reasoning"

    query = str(getattr(req, "query", "") or "").strip()
    lowered = query.lower()

    if any(hint in lowered for hint in _RETRIEVAL_HINTS):
        return "knowledge_lookup"
    if any(hint in lowered for hint in _REASONING_HINTS):
        return "reasoning"

    token_count = len(re.findall(r"\w+", query))
    if any(hint in lowered for hint in _SIMPLE_QUERY_HINTS):
        return "simple_text"
    if token_count <= 18 and len(query) <= 140 and "?" in query:
        return "simple_text"
    if token_count <= 12 and len(query) <= 100:
        return "simple_text"
    return "reasoning"


def _infer_retrieval_profile(query: str, workload: str) -> str:
    """Choose the cheapest retrieval depth that still matches the query intent."""
    lowered = str(query or "").lower()
    if workload == "simple_text":
        return "no_retrieval"
    if workload == "knowledge_lookup":
        if any(hint in lowered for hint in _SOURCE_REQUIRED_HINTS):
            return "full_retrieval"
        return "light_retrieval"
    if workload == "vision":
        return "full_retrieval"
    return "full_retrieval"


def _workload_sync_deadline_seconds(workload: str) -> int:
    """Return the synchronous deadline budget for one workload class."""
    key_map = {
        "simple_text": "SYNC_DEADLINE_SIMPLE_SECONDS",
        "knowledge_lookup": "SYNC_DEADLINE_KNOWLEDGE_SECONDS",
        "reasoning": "SYNC_DEADLINE_REASONING_SECONDS",
        "vision": "SYNC_DEADLINE_VISION_SECONDS",
    }
    default_map = {
        "simple_text": 25,
        "knowledge_lookup": 40,
        "reasoning": 70,
        "vision": 100,
    }
    key = key_map.get(workload, "SYNC_DEADLINE_REASONING_SECONDS")
    default = default_map.get(workload, 70)
    return max(5, int(settings.get(key, default)))


def _workload_provider_timeout_seconds(workload: str) -> int:
    """Return the provider timeout budget for one workload class."""
    key_map = {
        "simple_text": "PROVIDER_TIMEOUT_SIMPLE_SECONDS",
        "knowledge_lookup": "PROVIDER_TIMEOUT_KNOWLEDGE_SECONDS",
        "reasoning": "PROVIDER_TIMEOUT_REASONING_SECONDS",
        "vision": "PROVIDER_TIMEOUT_VISION_SECONDS",
    }
    default_map = {
        "simple_text": 20,
        "knowledge_lookup": 35,
        "reasoning": 60,
        "vision": 90,
    }
    key = key_map.get(workload, "PROVIDER_TIMEOUT_REASONING_SECONDS")
    default = default_map.get(workload, 60)
    return max(5, int(settings.get(key, default)))


def _effective_sync_timeout_seconds(req_timeout_seconds: int | None, runtime_profile: Dict[str, Any]) -> int:
    """Clamp request-level timeout overrides to the workload-specific sync deadline.

    Clients may send a generous `timeout_seconds`, but the runtime should not
    let simple interactive workloads inherit a slower budget than the profile
    selected for that workload class. This helper keeps lower client overrides
    intact while preventing higher overrides from stretching the synchronous
    path beyond the workload deadline.
    """
    runtime_deadline = int((runtime_profile.get("runtime_hints") or {}).get("sync_deadline_seconds", 0) or 0)
    if runtime_deadline <= 0:
        return max(5, int(req_timeout_seconds or settings.get("REQUEST_TIMEOUT_SECONDS", 120)))
    if req_timeout_seconds is None:
        return runtime_deadline
    return max(5, min(int(req_timeout_seconds), runtime_deadline))


def apply_query_runtime_profile(req: Any, modality: str, image_input: str | None) -> Dict[str, Any]:
    """Derive execution knobs for one request without mutating the incoming request object."""
    workload = classify_query_workload(req, modality=modality, image_input=image_input)
    perf_mode_enabled = str(settings.get("ROUTER_PERF_MODE", "0")).strip() == "1"
    simple_query_cap = int(settings.get("ROUTER_SIMPLE_QUERY_MAX_TOKENS", settings.MAX_TOKENS_DEFAULT))
    simple_text_max_fallbacks = max(1, int(settings.get("ROUTER_SIMPLE_TEXT_MAX_FALLBACKS", 1)))
    rag_simple_bypass = str(settings.get("RAG_SIMPLE_QUERY_BYPASS_ENABLED", "1")).strip() == "1"
    light_top_k = max(1, int(settings.get("RAG_LIGHT_TOP_K", 2)))
    light_context_budget = max(128, int(settings.get("RAG_LIGHT_CONTEXT_TOKEN_BUDGET", 480)))
    full_context_budget = max(light_context_budget, int(settings.get("RAG_FULL_CONTEXT_TOKEN_BUDGET", settings.get("RAG_CONTEXT_TOKEN_BUDGET", 1200))))
    rerank_for_light = str(settings.get("RERANK_ENABLED_FOR_LIGHT_RETRIEVAL", "0")).strip() == "1"

    use_rag = bool(req.enable_rag_for_answer or req.enable_rag_for_image)
    effective_max_tokens = req.max_tokens or settings.MAX_TOKENS_DEFAULT
    retrieval_mode = _infer_retrieval_profile(getattr(req, "query", ""), workload)
    top_k = max(1, light_top_k)
    context_token_budget = full_context_budget
    rerank_enabled = True
    max_fallbacks = 2
    needs_retrieval = retrieval_mode != "no_retrieval"
    needs_rerank = True
    interactive_priority = "normal"

    if perf_mode_enabled and workload == "simple_text":
        effective_max_tokens = min(int(effective_max_tokens), max(32, simple_query_cap))

    if workload == "simple_text" and rag_simple_bypass:
        use_rag = False
        retrieval_mode = "no_retrieval"
        top_k = 1
        context_token_budget = light_context_budget
        rerank_enabled = False
        needs_retrieval = False
        needs_rerank = False
        interactive_priority = "high"
        max_fallbacks = simple_text_max_fallbacks
    elif workload == "knowledge_lookup":
        use_rag = True if use_rag or perf_mode_enabled else use_rag
        retrieval_mode = _infer_retrieval_profile(getattr(req, "query", ""), workload)
        top_k = light_top_k
        context_token_budget = light_context_budget if retrieval_mode == "light_retrieval" else full_context_budget
        rerank_enabled = rerank_for_light if retrieval_mode == "light_retrieval" else True
        needs_retrieval = True
        needs_rerank = bool(rerank_enabled)
        interactive_priority = "high"
        max_fallbacks = 1 if perf_mode_enabled else 2
    elif workload == "reasoning":
        retrieval_mode = "full_retrieval" if use_rag else "no_retrieval"
        top_k = max(3, light_top_k + 1)
        context_token_budget = full_context_budget
        rerank_enabled = True
        needs_retrieval = retrieval_mode != "no_retrieval"
        needs_rerank = True
        interactive_priority = "normal"
        max_fallbacks = 2

    if modality in {"vision", "multimodal"} or image_input:
        use_rag = bool(req.enable_rag_for_answer or req.enable_rag_for_image)
        retrieval_mode = "full_retrieval" if use_rag else "no_retrieval"
        top_k = max(3, light_top_k + 1)
        context_token_budget = full_context_budget
        rerank_enabled = True
        needs_retrieval = retrieval_mode != "no_retrieval"
        needs_rerank = True
        interactive_priority = "high"
        max_fallbacks = 2

    detected_complexity = detect_query_complexity(str(getattr(req, "query", "") or ""), workload)
    runtime_hints: Dict[str, Any] = {
        "workload_class": workload,
        "retrieval_mode": retrieval_mode if use_rag else "no_retrieval",
        "rag_top_k": top_k,
        "rag_context_token_budget": context_token_budget,
        "rag_rerank_enabled": rerank_enabled,
        "max_fallbacks": max_fallbacks,
        "needs_retrieval": bool(use_rag and needs_retrieval),
        "needs_rerank": bool(use_rag and needs_rerank),
        "interactive_priority": interactive_priority,
        "provider_timeout_seconds": _workload_provider_timeout_seconds(workload),
        "sync_deadline_seconds": _workload_sync_deadline_seconds(workload),
    }
    adjusted = apply_complexity_runtime_adjustments(
        detected_complexity=detected_complexity,
        workload_class=workload,
        max_tokens=int(effective_max_tokens),
        sync_deadline_seconds=int(runtime_hints["sync_deadline_seconds"]),
        provider_timeout_seconds=int(runtime_hints["provider_timeout_seconds"]),
        runtime_hints=runtime_hints,
        workload_hints=getattr(req, "workload_hints", None),
    )
    effective_max_tokens = int(adjusted["max_tokens"])
    runtime_hints = adjusted["runtime_hints"]
    detected_complexity = adjusted["detected_complexity"]

    expected_tokens = None
    workload_hints = getattr(req, "workload_hints", None)
    if workload_hints is not None:
        expected_tokens = (
            workload_hints.get("expected_tokens")
            if isinstance(workload_hints, dict)
            else getattr(workload_hints, "expected_tokens", None)
        )
    if expected_tokens:
        effective_max_tokens = max(effective_max_tokens, int(expected_tokens))

    try:
        QUERY_COMPLEXITY_DETECTED.labels(
            detected_complexity=detected_complexity,
            workload_class=workload,
        ).inc()
    except Exception:
        pass

    return {
        "workload_class": workload,
        "detected_complexity": detected_complexity,
        "perf_mode_enabled": perf_mode_enabled,
        "use_rag": use_rag,
        "max_tokens": effective_max_tokens,
        "runtime_hints": runtime_hints,
    }
