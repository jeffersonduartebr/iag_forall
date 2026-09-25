# Objective: Error budget per provider, so one failing upstream is routed around instead of forcing local-only.
"""Rolling failure rate per provider (``ollama``, ``gemini``, ``openrouter``...), fed by every provider attempt.

The global error budget answered "the router is failing" by restricting candidates to local models. When the
failures came from the local GPU itself (timeouts under load), that locked the system into the failing
provider (production, 2026-09-24: 89 of 100 queries in 502). Tracking the rate per provider lets the router
drop exactly the provider that is failing and keep the others.

One Redis hash per 10 s bucket holds every provider's counters (``<provider>:total``/``<provider>:errors``),
so a check costs the same bucket scan as the global budget, whatever the number of providers.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, Set

from ..utils.background import spawn
from ..utils.redis_async_ops import redis_hgetall_map, redis_pipeline_execute
from .router_resilience import error_budget_window, safe_setting_bool

Getter = Callable[[str, object], object]
_PREFIX = "router:provider_budget:"


def provider_of(model: str) -> str:
    """Route prefix of a router model name (``openrouter/deepseek/x`` -> ``openrouter``)."""
    return str(model or "").split("/", 1)[0]


def record_provider_outcome(settings_getter: Getter, model: str, success: bool) -> None:
    """Count one provider attempt in the background (never blocks nor fails the request)."""
    if not safe_setting_bool(settings_getter, "ERROR_BUDGET_ENABLED", True):
        return
    try:
        spawn(_record(settings_getter, provider_of(model), success), name="provider_budget", limit=256)
    except RuntimeError:
        pass  # sem event loop (caminho síncrono de teste): a amostra se perde, o pedido não


async def _record(settings_getter: Getter, provider: str, success: bool) -> None:
    key = f"{_PREFIX}{int(time.time()) // 10}"
    window_s, _, _ = error_budget_window(settings_getter)

    def _build(pipe):
        pipe.hincrby(key, f"{provider}:total", 1)
        if not success:
            pipe.hincrby(key, f"{provider}:errors", 1)
        pipe.expire(key, window_s + 60)

    try:
        await redis_pipeline_execute(_build)
    except Exception:
        pass


def _sum_counters(buckets: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    totals: Dict[str, Dict[str, int]] = {}
    for raw in buckets:
        for field, value in raw.items():
            provider, _, counter = str(field).rpartition(":")
            slot = totals.setdefault(provider, {"total": 0, "errors": 0})
            slot[counter] = slot.get(counter, 0) + int(value or 0)
    return totals


async def providers_over_budget(settings_getter: Getter) -> Set[str]:
    """Providers whose failure rate in the window reached the threshold (with enough samples)."""
    if not safe_setting_bool(settings_getter, "ERROR_BUDGET_ENABLED", True):
        return set()
    window_s, threshold, min_requests = error_budget_window(settings_getter)
    now = int(time.time()) // 10
    try:
        buckets = [await redis_hgetall_map(f"{_PREFIX}{now - offset}") for offset in range(window_s // 10 + 1)]
    except Exception:
        return set()
    return {
        provider
        for provider, c in _sum_counters(buckets).items()
        if c["total"] >= min_requests and c["errors"] / c["total"] >= threshold
    }
