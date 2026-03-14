# -*- coding: utf-8 -*-
# Objective: HTTP middleware for adaptive admission control and overload-aware request limiting.
"""Apply adaptive request admission based on current Ollama pressure.

The original middleware enforced a fixed sliding-window limit per client IP,
which was too blunt for local inference traffic and synthetic load tests. This
module keeps the sliding-window store, but uses it only when the local provider
is under pressure. Requests are segmented by tenant first, route class second,
and IP only as a fallback identity.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.middleware.backpressure import get_backpressure
from app.observability import (
    ADAPTIVE_LIMITER_IDENTITY_BUCKETS,
    ADAPTIVE_LIMITER_OVERLOAD_EVENTS,
    ADAPTIVE_LIMITER_REJECTIONS,
    ADAPTIVE_LIMITER_STATE,
    ROUTER_ENQUEUED_DUE_TO_DEADLINE,
    ROUTER_ENQUEUED_DUE_TO_QUEUE_WAIT,
)
from app.providers_async import get_ollama_admission_snapshot
from app.settings_dynamic import settings

logger = logging.getLogger(__name__)

_redis_client = None


def _get_redis():
    """Return a shared Redis client when available."""
    global _redis_client
    try:
        from app.utils.redis_client import ensure_redis_connected, get_redis_async_safe

        _redis_client = get_redis_async_safe() or ensure_redis_connected(max_wait_s=0.0, min_retry_interval_s=2.0)
    except Exception:
        _redis_client = None
    return _redis_client


def _as_bool(key: str, default: bool) -> bool:
    """Read one runtime boolean setting with tolerant string parsing."""
    return str(settings.get(key, "1" if default else "0")).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(key: str, default: int, minimum: int = 0) -> int:
    """Read one runtime integer setting with bounds."""
    try:
        return max(minimum, int(settings.get(key, default)))
    except Exception:
        return max(minimum, int(default))


def _as_float(key: str, default: float, minimum: float = 0.0, maximum: float | None = None) -> float:
    """Read one runtime float setting with bounds."""
    try:
        value = float(settings.get(key, default))
    except Exception:
        value = float(default)
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _adaptive_limiter_config() -> Dict[str, float | int | bool]:
    """Return the current adaptive-admission settings from the dynamic catalog."""
    return {
        "enabled": _as_bool("ADAPTIVE_LIMITER_ENABLED", True),
        "window_seconds": _as_int("ADAPTIVE_LIMITER_WINDOW_SECONDS", 15, minimum=1),
        "hysteresis_windows": _as_int("ADAPTIVE_LIMITER_HYSTERESIS_WINDOWS", 3, minimum=1),
        "elevated_utilization": _as_float("ADAPTIVE_LIMITER_ELEVATED_UTILIZATION", 0.80, minimum=0.1, maximum=1.5),
        "congested_utilization": _as_float("ADAPTIVE_LIMITER_CONGESTED_UTILIZATION", 1.00, minimum=0.1, maximum=2.0),
        "elevated_queue_wait_ms": _as_float("ADAPTIVE_LIMITER_ELEVATED_QUEUE_WAIT_P95_MS", 500.0, minimum=50.0),
        "congested_queue_wait_ms": _as_float("ADAPTIVE_LIMITER_CONGESTED_QUEUE_WAIT_P95_MS", 1000.0, minimum=100.0),
        "interactive_per_slot_elevated": _as_int("ADAPTIVE_LIMITER_INTERACTIVE_PER_SLOT_ELEVATED", 12, minimum=1),
        "interactive_per_slot_congested": _as_int("ADAPTIVE_LIMITER_INTERACTIVE_PER_SLOT_CONGESTED", 6, minimum=1),
        "admin_per_slot_elevated": _as_int("ADAPTIVE_LIMITER_ADMIN_PER_SLOT_ELEVATED", 3, minimum=1),
        "admin_per_slot_congested": _as_int("ADAPTIVE_LIMITER_ADMIN_PER_SLOT_CONGESTED", 1, minimum=1),
        "sync_queue_wait_ms": _as_float("ADAPTIVE_LIMITER_SYNC_QUEUE_WAIT_MS", 250.0, minimum=50.0),
    }


class RateLimitStore:
    """Sliding-window store backed by Redis when available and memory otherwise."""

    REDIS_PREFIX = "adaptive-limit:"
    REDIS_TTL = 3600
    ENTRY_TTL_SECONDS = 3600
    CLEANUP_INTERVAL_SECONDS = 300

    def __init__(self) -> None:
        """Initialize the mutable backend state for adaptive limiter buckets."""
        self._memory_store: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
        self._use_redis: Optional[bool] = None
        self._next_redis_probe_at = 0.0
        self._redis_reprobe_interval_s = 10.0

    def _publish_bucket_metrics(self) -> None:
        """Expose a rough count of active identity buckets for observability."""
        backend = "redis" if self._use_redis else "memory"
        try:
            ADAPTIVE_LIMITER_IDENTITY_BUCKETS.labels(backend=backend).set(len(self._memory_store))
        except Exception:
            pass

    def _should_use_redis(self) -> bool:
        """Probe Redis lazily and fall back to local state on failure."""
        now = time.time()
        should_probe = self._use_redis is None or (self._use_redis is False and now >= self._next_redis_probe_at)
        if should_probe:
            rds = _get_redis()
            self._use_redis = rds is not None
            if self._use_redis:
                self._next_redis_probe_at = 0.0
                logger.info("[adaptive_limiter] Using Redis backend")
            else:
                self._next_redis_probe_at = now + self._redis_reprobe_interval_s
                logger.warning("[adaptive_limiter] Redis unavailable, using in-memory backend")
        return bool(self._use_redis)

    async def is_rate_limited(self, scope_key: str, max_requests: int, window_seconds: int) -> bool:
        """Return whether one identity bucket has exhausted its quota."""
        if max_requests <= 0:
            return True
        if self._should_use_redis():
            return await self._is_rate_limited_redis(scope_key, max_requests, window_seconds)
        return await self._is_rate_limited_memory(scope_key, max_requests, window_seconds)

    async def _is_rate_limited_redis(self, scope_key: str, max_requests: int, window_seconds: int) -> bool:
        """Use Redis sorted sets to enforce a distributed sliding window."""
        rds = _get_redis()
        if rds is None:
            self._use_redis = False
            self._next_redis_probe_at = time.time() + self._redis_reprobe_interval_s
            return await self._is_rate_limited_memory(scope_key, max_requests, window_seconds)
        try:
            key = f"{self.REDIS_PREFIX}{scope_key}"
            now = time.time()
            cutoff = now - window_seconds
            pipe = rds.pipeline()
            pipe.zremrangebyscore(key, 0, cutoff)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, self.REDIS_TTL)
            results = pipe.execute()
            count = int(results[1])
            self._publish_bucket_metrics()
            return count >= max_requests
        except Exception as exc:
            logger.warning("[adaptive_limiter] Redis error, falling back to memory: %s", exc)
            self._use_redis = False
            self._next_redis_probe_at = time.time() + self._redis_reprobe_interval_s
            return await self._is_rate_limited_memory(scope_key, max_requests, window_seconds)

    async def _is_rate_limited_memory(self, scope_key: str, max_requests: int, window_seconds: int) -> bool:
        """Use a local sliding-window bucket per identity and route class."""
        now = time.time()
        cutoff = now - window_seconds
        async with self._lock:
            bucket = self._memory_store.setdefault(scope_key, [])
            bucket[:] = [ts for ts in bucket if ts > cutoff]
            if len(bucket) >= max_requests:
                self._publish_bucket_metrics()
                return True
            bucket.append(now)
            self._publish_bucket_metrics()
            return False

    async def cleanup(self) -> int:
        """Trim stale in-memory buckets to prevent unbounded growth."""
        now = time.time()
        cutoff = now - self.ENTRY_TTL_SECONDS
        cleaned = 0
        async with self._lock:
            for key in list(self._memory_store.keys()):
                before = len(self._memory_store[key])
                self._memory_store[key] = [ts for ts in self._memory_store[key] if ts > cutoff]
                cleaned += before - len(self._memory_store[key])
                if not self._memory_store[key]:
                    del self._memory_store[key]
            self._publish_bucket_metrics()
        return cleaned

    def get_stats(self) -> dict:
        """Return small diagnostic statistics about the current backend."""
        return {
            "using_redis": self._use_redis,
            "memory_entries": len(self._memory_store),
            "memory_total_requests": sum(len(v) for v in self._memory_store.values()),
        }


class PressureStateTracker:
    """Apply simple hysteresis so admission state does not flap on short bursts."""

    _VALUE_MAP = {"normal": 0, "elevated": 1, "congested": 2}

    def __init__(self) -> None:
        self._state = "normal"
        self._candidate_state = "normal"
        self._candidate_windows = 0
        self._lock = asyncio.Lock()

    async def update(self, candidate_state: str, hysteresis_windows: int) -> str:
        """Transition to a new pressure state only after repeated observations."""
        async with self._lock:
            if candidate_state == self._state:
                self._candidate_state = candidate_state
                self._candidate_windows = 0
                self._publish()
                return self._state
            if candidate_state != self._candidate_state:
                self._candidate_state = candidate_state
                self._candidate_windows = 1
            else:
                self._candidate_windows += 1
            if self._candidate_windows >= max(1, hysteresis_windows):
                self._state = candidate_state
                self._candidate_windows = 0
                ADAPTIVE_LIMITER_OVERLOAD_EVENTS.labels(pressure_state=self._state).inc()
            self._publish()
            return self._state

    def _publish(self) -> None:
        """Write the current state to Prometheus."""
        try:
            ADAPTIVE_LIMITER_STATE.set(self._VALUE_MAP.get(self._state, 0))
        except Exception:
            pass


rate_limit_store = RateLimitStore()
_pressure_tracker = PressureStateTracker()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Throttle requests only when the local provider is under real pressure."""

    EXEMPT_PATHS = frozenset(
        [
            "/health",
            "/healthz",
            "/ready",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/v1/health",
        ]
    )
    QUERY_PATHS = frozenset(["/query", "/query/stream", "/v1/query"])
    TENANT_HEADERS = ("X-Tenant-ID", "X-Tenant", "X-School-ID")

    async def dispatch(self, request: Request, call_next):
        """Admit or reject one request based on runtime pressure and route class."""
        route_class = self._classify_route(request.url.path)
        if route_class == "health_observability":
            return await call_next(request)

        cfg = _adaptive_limiter_config()
        if not bool(cfg["enabled"]):
            return await call_next(request)

        snapshot = get_ollama_admission_snapshot()
        candidate_state = self._candidate_pressure_state(snapshot, cfg)
        pressure_state = await _pressure_tracker.update(candidate_state, int(cfg["hysteresis_windows"]))

        # When the provider is healthy, the adaptive limiter does not cap normal traffic.
        if pressure_state == "normal":
            response = await call_next(request)
            response.headers["X-Admission-State"] = pressure_state
            response.headers["X-RateLimit-Scope"] = route_class
            return response

        if route_class == "interactive_query" and self._should_preempt_to_async(snapshot, pressure_state, cfg):
            request.state.defer_to_query_job = True
            request.state.query_job_reason = "ollama_queue_wait"
            request.state.query_job_pressure_state = pressure_state
            request.state.query_job_scope = route_class
            request.state.query_job_workload_class = "unknown"
            try:
                ROUTER_ENQUEUED_DUE_TO_QUEUE_WAIT.labels(pressure_state=pressure_state).inc()
                ROUTER_ENQUEUED_DUE_TO_DEADLINE.labels(workload_class="unknown").inc()
            except Exception:
                pass
            response = await call_next(request)
            response.headers["X-Admission-State"] = pressure_state
            response.headers["X-RateLimit-Scope"] = route_class
            response.headers["X-RateLimit-Reason"] = "ollama_queue_wait"
            return response

        identity, identity_type = await self._resolve_identity(request)
        quota = self._quota_for(route_class, pressure_state, snapshot, cfg)
        if quota is not None:
            scope_key = f"{identity_type}:{identity}:{route_class}:{pressure_state}"
            limited = await rate_limit_store.is_rate_limited(
                scope_key,
                max_requests=quota,
                window_seconds=int(cfg["window_seconds"]),
            )
            if limited:
                reason = "ollama_overloaded"
                if route_class == "interactive_query":
                    request.state.defer_to_query_job = True
                    request.state.query_job_reason = reason
                    request.state.query_job_pressure_state = pressure_state
                    request.state.query_job_scope = route_class
                    request.state.query_job_workload_class = "unknown"
                    response = await call_next(request)
                    response.headers["X-Admission-State"] = pressure_state
                    response.headers["X-RateLimit-Scope"] = route_class
                    response.headers["X-RateLimit-Reason"] = reason
                    response.headers["X-RateLimit-Limit"] = str(quota)
                    response.headers["X-RateLimit-Window"] = str(int(cfg["window_seconds"]))
                    return response
                ADAPTIVE_LIMITER_REJECTIONS.labels(
                    route_class=route_class,
                    identity_type=identity_type,
                    reason=reason,
                    pressure_state=pressure_state,
                ).inc()
                retry_after = 5 if pressure_state == "congested" else 2
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": True,
                        "message": "Rate limit exceeded. Please try again later.",
                        "retry_after": retry_after,
                        "reason": reason,
                        "pressure_state": pressure_state,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-Admission-State": pressure_state,
                        "X-RateLimit-Reason": reason,
                        "X-RateLimit-Scope": route_class,
                        "X-RateLimit-Limit": str(quota),
                        "X-RateLimit-Window": str(int(cfg["window_seconds"])),
                    },
                )

        response = await call_next(request)
        response.headers["X-Admission-State"] = pressure_state
        response.headers["X-RateLimit-Scope"] = route_class
        if quota is not None:
            response.headers["X-RateLimit-Limit"] = str(quota)
            response.headers["X-RateLimit-Window"] = str(int(cfg["window_seconds"]))
        return response

    def _should_preempt_to_async(
        self,
        snapshot: Dict[str, float | int | str],
        pressure_state: str,
        cfg: Dict[str, float | int | bool],
    ) -> bool:
        """Return whether interactive traffic should skip the sync path immediately."""
        if pressure_state == "normal":
            return False
        current_limit = max(1, int(snapshot.get("current_limit", 1) or 1))
        total_inflight = int(snapshot.get("total_inflight", 0) or 0)
        queue_wait_ms = float(snapshot.get("max_queue_wait_ms", 0.0) or 0.0)
        utilization = float(snapshot.get("utilization", 0.0) or 0.0)
        if queue_wait_ms >= float(cfg["sync_queue_wait_ms"]):
            return True
        if total_inflight >= current_limit:
            return True
        if pressure_state == "congested" and utilization >= float(cfg["elevated_utilization"]):
            return True
        return False

    def _classify_route(self, path: str) -> str:
        """Map concrete paths to a small set of policy classes."""
        if path in self.EXEMPT_PATHS:
            return "health_observability"
        if path in self.QUERY_PATHS:
            return "interactive_query"
        if path.startswith("/admin") or path.startswith("/feedback") or path.startswith("/ops"):
            return "admin_eval_governance"
        return "admin_eval_governance"

    async def _resolve_identity(self, request: Request) -> Tuple[str, str]:
        """Resolve tenant-first identity for adaptive-limiter buckets."""
        for header in self.TENANT_HEADERS:
            value = (request.headers.get(header) or "").strip()
            if value:
                return value[:128], "tenant"
        tenant_query = (request.query_params.get("tenant_id") or "").strip()
        if tenant_query:
            return tenant_query[:128], "tenant"
        return self._client_ip(request), "ip"

    def _client_ip(self, request: Request) -> str:
        """Resolve the client IP, honoring the first X-Forwarded-For value."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            first = forwarded_for.split(",")[0].strip()
            if first:
                return first
        return request.client.host if request.client else "unknown"

    def _candidate_pressure_state(self, snapshot: Dict[str, float | int | str], cfg: Dict[str, float | int | bool]) -> str:
        """Classify the current runtime pressure before hysteresis is applied."""
        utilization = float(snapshot.get("utilization", 0.0) or 0.0)
        queue_wait_ms = float(snapshot.get("max_queue_wait_ms", 0.0) or 0.0)
        pressure_state = str(snapshot.get("pressure_state", "normal") or "normal")
        backpressure = get_backpressure()
        bp_stats = backpressure.get_stats()
        bp_utilization = float(bp_stats.get("utilization", 0.0) or 0.0)

        if pressure_state == "congested":
            return "congested"
        if (
            utilization >= float(cfg["congested_utilization"])
            or queue_wait_ms >= float(cfg["congested_queue_wait_ms"])
            or bp_utilization >= 0.95
        ):
            return "congested"
        if (
            pressure_state == "elevated"
            or utilization >= float(cfg["elevated_utilization"])
            or queue_wait_ms >= float(cfg["elevated_queue_wait_ms"])
            or bp_utilization >= 0.85
        ):
            return "elevated"
        return "normal"

    def _quota_for(
        self,
        route_class: str,
        pressure_state: str,
        snapshot: Dict[str, float | int | str],
        cfg: Dict[str, float | int | bool],
    ) -> Optional[int]:
        """Compute a short-window quota for one route class under overload."""
        if pressure_state == "normal" or route_class == "health_observability":
            return None
        current_limit = max(1, int(snapshot.get("current_limit", 1) or 1))
        if route_class == "interactive_query":
            per_slot = int(
                cfg["interactive_per_slot_congested"]
                if pressure_state == "congested"
                else cfg["interactive_per_slot_elevated"]
            )
            return max(per_slot, current_limit * per_slot)
        per_slot = int(
            cfg["admin_per_slot_congested"]
            if pressure_state == "congested"
            else cfg["admin_per_slot_elevated"]
        )
        return max(1, current_limit * per_slot)


async def periodic_cleanup() -> None:
    """Periodically trim stale identity buckets from the local fallback backend."""
    while True:
        await asyncio.sleep(rate_limit_store.CLEANUP_INTERVAL_SECONDS)
        try:
            cleaned = await rate_limit_store.cleanup()
            if cleaned > 0:
                logger.debug("[adaptive_limiter] Cleaned up %s old entries", cleaned)
        except Exception as exc:
            logger.warning("[adaptive_limiter] Cleanup failed: %s", exc)
