# Objective: Service-layer helpers for router resilience.
"""Resilience helpers extracted from router_core."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Tuple, cast

import pybreaker

from ..observability import DEPENDENCY_CIRCUIT_STATE, DEPENDENCY_FAILURES
from ..utils.redis_async_ops import redis_hgetall_map, redis_pipeline_execute
from ..utils.redis_client import ensure_redis_connected, get_redis_async_safe


def safe_setting_int(settings_getter: Callable[[str, int], object], key: str, default: int) -> int:
    """Read one integer setting defensively."""
    try:
        return int(cast(Any, settings_getter(key, default)))
    except Exception:
        return int(default)


def safe_setting_float(settings_getter: Callable[[str, float], object], key: str, default: float) -> float:
    """Read one float setting defensively."""
    try:
        return float(cast(Any, settings_getter(key, default)))
    except Exception:
        return float(default)


def safe_setting_bool(settings_getter: Callable[[str, str], object], key: str, default: bool) -> bool:
    """Read one boolean setting defensively."""
    raw_default = "1" if default else "0"
    try:
        raw = str(settings_getter(key, raw_default)).strip()
    except Exception:
        raw = raw_default
    return raw in ("1", "true", "True")


def get_router_redis():
    """Resolve the shared Redis client used by router resilience helpers."""
    return get_redis_async_safe() or ensure_redis_connected(max_wait_s=0.0, min_retry_interval_s=2.0)


dep_cache_breaker = pybreaker.CircuitBreaker(fail_max=10, reset_timeout=30, name="dep_cache")
dep_uq_breaker = pybreaker.CircuitBreaker(fail_max=10, reset_timeout=30, name="dep_uq")


def record_dependency_breaker_metrics() -> None:
    """Publish breaker states as stable numeric metrics."""
    mapping = {"closed": 0, "half-open": 1, "open": 2}
    try:
        DEPENDENCY_CIRCUIT_STATE.labels(dependency="cache").set(mapping.get(dep_cache_breaker.current_state, 0))
        DEPENDENCY_CIRCUIT_STATE.labels(dependency="uq").set(mapping.get(dep_uq_breaker.current_state, 0))
    except Exception:
        pass


def error_budget_window(settings_getter: Callable[[str, object], object]) -> Tuple[int, float, int]:
    """Return the active error budget policy."""
    window_s = safe_setting_int(settings_getter, "ERROR_BUDGET_WINDOW_S", 300)
    threshold = safe_setting_float(settings_getter, "ERROR_BUDGET_THRESHOLD", 0.20)
    min_requests = safe_setting_int(settings_getter, "ERROR_BUDGET_MIN_REQUESTS", 20)
    return max(10, window_s), max(0.0, min(1.0, threshold)), max(1, min_requests)


def record_request_outcome(*, settings_getter: Callable[[str, object], object], success: bool) -> None:
    """Record one success/failure sample into the rolling error budget."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(record_request_outcome_async(settings_getter=settings_getter, success=success))
    except RuntimeError:
        _record_request_outcome_sync(settings_getter=settings_getter, success=success)


async def record_request_outcome_async(*, settings_getter: Callable[[str, object], object], success: bool) -> None:
    """Record one error-budget sample without blocking the event loop."""
    if not safe_setting_bool(settings_getter, "ERROR_BUDGET_ENABLED", True):
        return
    try:
        now = int(time.time())
        bucket = now // 10
        window_s, _, _ = error_budget_window(settings_getter)
        key = f"router:error_budget:{bucket}"

        def _build(pipe):
            pipe.hincrby(key, "total", 1)
            if not success:
                pipe.hincrby(key, "errors", 1)
            pipe.expire(key, window_s + 60)

        await redis_pipeline_execute(_build)
    except Exception:
        pass


def _record_request_outcome_sync(*, settings_getter: Callable[[str, object], object], success: bool) -> None:
    rds = get_router_redis()
    if not rds or not safe_setting_bool(settings_getter, "ERROR_BUDGET_ENABLED", True):
        return
    try:
        now = int(time.time())
        bucket = now // 10
        window_s, _, _ = error_budget_window(settings_getter)
        key = f"router:error_budget:{bucket}"
        pipe = rds.pipeline()
        pipe.hincrby(key, "total", 1)
        if not success:
            pipe.hincrby(key, "errors", 1)
        pipe.expire(key, window_s + 60)
        pipe.execute()
    except Exception:
        pass


def is_error_budget_exceeded(*, settings_getter: Callable[[str, object], object]) -> bool:
    """Return whether the rolling error budget is currently exceeded."""
    rds = get_router_redis()
    if not rds or not safe_setting_bool(settings_getter, "ERROR_BUDGET_ENABLED", True):
        return False
    try:
        window_s, threshold, min_requests = error_budget_window(settings_getter)
        now = int(time.time())
        total = 0
        errors = 0
        for offset in range((window_s // 10) + 1):
            bucket = (now // 10) - offset
            key = f"router:error_budget:{bucket}"
            raw = rds.hgetall(key) or {}
            total += int(raw.get(b"total") or raw.get("total") or 0)
            errors += int(raw.get(b"errors") or raw.get("errors") or 0)
        if total < min_requests:
            return False
        return (errors / max(1, total)) >= threshold
    except Exception:
        DEPENDENCY_FAILURES.labels(dependency="error_budget").inc()
        return False


async def is_error_budget_exceeded_async(*, settings_getter: Callable[[str, object], object]) -> bool:
    """Async error-budget check for request hot paths."""
    if not safe_setting_bool(settings_getter, "ERROR_BUDGET_ENABLED", True):
        return False
    try:
        window_s, threshold, min_requests = error_budget_window(settings_getter)
        now = int(time.time())
        total = 0
        errors = 0
        for offset in range((window_s // 10) + 1):
            bucket = (now // 10) - offset
            key = f"router:error_budget:{bucket}"
            raw = await redis_hgetall_map(key)
            total += int(raw.get("total") or 0)
            errors += int(raw.get("errors") or 0)
        if total < min_requests:
            return False
        return (errors / max(1, total)) >= threshold
    except Exception:
        DEPENDENCY_FAILURES.labels(dependency="error_budget").inc()
        return False
