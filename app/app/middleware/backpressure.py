# -*- coding: utf-8 -*-
# Objective: HTTP middleware for backpressure.
"""
backpressure.py — Global Concurrency Limit Middleware
------------------------------------------------------
Provides backpressure control to prevent system overload by limiting
the number of concurrent requests being processed.

When the limit is reached, new requests receive HTTP 503 Service Unavailable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.observability import (
    BACKPRESSURE_QUEUE_SIZE,
    BACKPRESSURE_REJECTED,
)
from app.settings_dynamic import settings
from app.utils.redis_distributed import RedisGlobalSemaphore

logger = logging.getLogger(__name__)


def _redis_backpressure_enabled() -> bool:
    return str(settings.get("BACKPRESSURE_REDIS_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}


class BackpressureSemaphore:
    """
    Global semaphore for request concurrency control.

    Thread-safe singleton that tracks concurrent requests.
    """

    _instance: Optional["BackpressureSemaphore"] = None
    _lock = asyncio.Lock()
    _initialized: bool = False

    def __new__(cls) -> "BackpressureSemaphore":
        """Return the instance created for this class.

This hook is typically used to enforce singleton-style behavior or other allocation constraints."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
        if self._initialized:
            return

        self._max_concurrent = settings.MAX_CONCURRENT_REQUESTS
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._current_count = 0
        self._count_lock = asyncio.Lock()
        self._initialized = True
        logger.info(f"[Backpressure] Initialized with max_concurrent={self._max_concurrent}")

    async def acquire(self) -> bool:
        """
        Try to acquire a slot for processing.

        Returns:
            True if slot acquired, False if at capacity
        """
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=0.001)
            async with self._count_lock:
                self._current_count += 1
                BACKPRESSURE_QUEUE_SIZE.set(self._current_count)
            return True
        except (asyncio.TimeoutError, Exception):
            return False

    async def release(self):
        """Release a processing slot."""
        try:
            async with self._count_lock:
                if self._current_count > 0:
                    self._current_count -= 1
                    BACKPRESSURE_QUEUE_SIZE.set(self._current_count)
            self._semaphore.release()
        except Exception:
            logger.warning("[Backpressure] Failed to release semaphore slot", exc_info=True)

    @property
    def current_load(self) -> int:
        """Get current number of in-flight requests."""
        return self._current_count

    @property
    def max_concurrent(self) -> int:
        """Get maximum concurrent requests allowed."""
        return self._max_concurrent

    def get_stats(self) -> dict:
        """Get backpressure statistics."""
        return {
            "current_load": self._current_count,
            "max_concurrent": self._max_concurrent,
            "utilization": self._current_count / self._max_concurrent if self._max_concurrent > 0 else 0,
        }


# Global instance
_backpressure: Optional[BackpressureSemaphore] = None


def get_backpressure() -> BackpressureSemaphore:
    """Get or create the global backpressure semaphore."""
    global _backpressure
    if _backpressure is None:
        _backpressure = BackpressureSemaphore()
    return _backpressure


class BackpressureMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces global concurrency limits.

    Returns HTTP 503 when the system is at capacity.
    """

    # Paths that bypass backpressure (health checks, metrics)
    BYPASS_PATHS = {"/health", "/healthz", "/ready", "/metrics"}
    QUERY_PATHS = {"/query", "/query/stream", "/v1/query", "/v1/chat/completions"}

    async def dispatch(self, request: Request, call_next) -> Response:
        if not settings.BACKPRESSURE_ENABLED:
            return await call_next(request)

        if request.url.path in self.BYPASS_PATHS:
            return await call_next(request)

        if getattr(request.state, "defer_to_query_job", False):
            return await call_next(request)

        redis_sem: RedisGlobalSemaphore | None = None
        acquired = False
        backpressure = None

        if _redis_backpressure_enabled():
            limit = max(1, int(settings.MAX_CONCURRENT_REQUESTS))
            redis_sem = RedisGlobalSemaphore("global-api", limit)
            acquired = await redis_sem.acquire()
        else:
            backpressure = get_backpressure()
            acquired = await backpressure.acquire()

        if not acquired:
            if request.url.path in self.QUERY_PATHS:
                request.state.defer_to_query_job = True
                request.state.query_job_reason = "backpressure"
                request.state.query_job_pressure_state = "congested"
                return await call_next(request)
            BACKPRESSURE_REJECTED.inc()
            logger.warning("[Backpressure] Rejecting request: at capacity")
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service temporarily unavailable",
                    "detail": "System at capacity, please retry later",
                    "retry_after": 5,
                },
                headers={"Retry-After": "5"},
            )

        try:
            return await call_next(request)
        finally:
            if redis_sem is not None:
                await redis_sem.release()
            elif backpressure is not None:
                await backpressure.release()
