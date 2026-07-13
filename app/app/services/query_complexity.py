# Objective: Runtime query complexity detection for adaptive routing.
"""Detect query complexity at runtime without client-provided difficulty labels."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

_EXPERT_HINTS = (
    "system design",
    "distributed lock",
    "fencing token",
    "saga",
    "idempotent",
    "linearizab",
    "serializab",
    "raft",
    "consensus",
    "mvcc",
    "crdt",
    "stride",
    "threat model",
    "zero-trust",
    "supply-chain",
    "exactly-once",
    "event sourcing",
    "cqrs",
    "formalize",
    "prove correctness",
    "multi-region",
    "active-active",
    "sharding",
    "rebalancing",
    "chaos engineering",
    "write skew",
    "split-brain",
    "thundering herd",
    "outbox pattern",
    "debezium",
    "litmus-test",
    "p99",
    "p95",
    "50k tps",
    "1m concurrent",
    "2b linhas",
    "gdpr-compliant",
    "crypto-shredding",
)

_HIGH_HINTS = (
    "compare and contrast",
    "compare ",
    "design ",
    "arquitete",
    "projete",
    "optimize",
    "otimize",
    "refactor",
    "migre",
    "implement ",
    "step by step",
    "passo a passo",
    "detalhadamente",
    "completo",
    "análise detalhada",
    "analise detalhada",
    "desenvolva uma solução",
    "explain how",
    "explique como",
    "trade-off",
    "trade off",
    "root-cause",
    "causa raiz",
)

_COMPLEXITY_ORDER = ("simple", "moderate", "high", "expert")

_COMPLEXITY_TIMEOUT_MULTIPLIER = {
    "simple": 1.0,
    "moderate": 1.05,
    "high": 1.35,
    "expert": 1.75,
}

_COMPLEXITY_MIN_MAX_TOKENS = {
    "simple": 256,
    "moderate": 512,
    "high": 2048,
    "expert": 4096,
}

_COMPLEXITY_SYNC_DEADLINE_BONUS = {
    "simple": 0,
    "moderate": 5,
    "high": 25,
    "expert": 45,
}

_VALID_COMPLEXITIES = frozenset(_COMPLEXITY_ORDER)


def detect_query_complexity(query: str, workload_class: str = "reasoning") -> str:
    """Infer complexity tier from query text and workload class only."""
    text = str(query or "").strip()
    if not text:
        return "simple"

    lowered = text.lower()
    tokens = len(re.findall(r"\w+", text, flags=re.UNICODE))
    chars = len(text)

    expert_score = sum(1 for hint in _EXPERT_HINTS if hint in lowered)
    high_score = sum(1 for hint in _HIGH_HINTS if hint in lowered)

    if tokens >= 80 or chars >= 500:
        high_score += 2
    if tokens >= 120 or chars >= 800:
        expert_score += 2
    if lowered.count("?") >= 2:
        high_score += 1
    if any(marker in lowered for marker in (" vs ", " versus ", " and ", " e também", " além de ")):
        high_score += 1
    if re.search(r"\b\d{2,}\s*(tps|rps|qps|usuários|users|linhas|rows)\b", lowered):
        expert_score += 1

    if expert_score >= 2 or (expert_score >= 1 and tokens >= 60):
        return "expert"
    if high_score >= 2 or tokens >= 50 or chars >= 350:
        return "high"
    if tokens <= 15 and chars <= 120 and workload_class == "simple_text":
        return "simple"
    if tokens <= 28 and chars <= 220:
        return "moderate"
    if workload_class in {"reasoning", "vision"}:
        return "high"
    return "moderate"


def apply_complexity_runtime_adjustments(
    *,
    detected_complexity: str,
    workload_class: str,
    max_tokens: int,
    sync_deadline_seconds: int,
    provider_timeout_seconds: int,
    runtime_hints: Dict[str, Any],
    workload_hints: Optional[Any] = None,
) -> Dict[str, Any]:
    """Tune runtime knobs based on detected complexity."""
    complexity = detected_complexity if detected_complexity in _VALID_COMPLEXITIES else "moderate"
    multiplier = _COMPLEXITY_TIMEOUT_MULTIPLIER.get(complexity, 1.0)
    min_tokens = _COMPLEXITY_MIN_MAX_TOKENS.get(complexity, 512)
    deadline_bonus = _COMPLEXITY_SYNC_DEADLINE_BONUS.get(complexity, 0)

    effective_max_tokens = max(int(max_tokens), min_tokens)
    effective_sync_deadline = int(sync_deadline_seconds * multiplier) + deadline_bonus
    effective_provider_timeout = int(provider_timeout_seconds * multiplier) + max(0, deadline_bonus - 5)

    theme = None
    benchmark_id = None
    if workload_hints is not None:
        if isinstance(workload_hints, dict):
            theme = workload_hints.get("theme")
            benchmark_id = workload_hints.get("benchmark_id")
        else:
            theme = getattr(workload_hints, "theme", None)
            benchmark_id = getattr(workload_hints, "benchmark_id", None)

    prefer_cloud_models = complexity in {"high", "expert"}
    prefer_strong_judge = complexity in {"high", "expert"}
    interactive_priority = runtime_hints.get("interactive_priority", "normal")
    if complexity in {"high", "expert"}:
        interactive_priority = "normal"
        runtime_hints["max_fallbacks"] = max(int(runtime_hints.get("max_fallbacks", 2)), 3)

    runtime_hints.update(
        {
            "detected_complexity": complexity,
            "prefer_cloud_models": prefer_cloud_models,
            "prefer_strong_judge": prefer_strong_judge,
            "interactive_priority": interactive_priority,
            "provider_timeout_seconds": effective_provider_timeout,
            "sync_deadline_seconds": effective_sync_deadline,
            "benchmark_theme": theme,
            "benchmark_id": benchmark_id,
        }
    )

    return {
        "max_tokens": effective_max_tokens,
        "runtime_hints": runtime_hints,
        "detected_complexity": complexity,
    }
