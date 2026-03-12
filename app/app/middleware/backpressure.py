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
import time
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.settings_dynamic import settings
from app.observability import (
    BACKPRESSURE_REJECTED,
    BACKPRESSURE_QUEUE_SIZE,
)

logger = logging.getLogger(__name__)


class BackpressureSemaphore:
    """
    Global semaphore for request concurrency control.

    Thread-safe singleton that tracks concurrent requests.
    """

    _instance: Optional["BackpressureSemaphore"] = None
    _lock = asyncio.Lock()

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
            async with self._count_lock:
                available_slots = int(getattr(self._semaphore, "_value", 0))
                if available_slots <= 0 or self._current_count >= self._max_concurrent:
                    return False
                self._semaphore._value = available_slots - 1
                self._current_count += 1
                BACKPRESSURE_QUEUE_SIZE.set(self._current_count)
                return True
        except Exception:
            return True  # On error, allow request through

    async def release(self):
        """Release a processing slot."""
        try:
            async with self._count_lock:
                if self._current_count <= 0:
                    return
                self._current_count = max(0, self._current_count - 1)
                current_slots = int(getattr(self._semaphore, "_value", 0))
                if current_slots < self._max_concurrent:
                    self._semaphore._value = current_slots + 1
                BACKPRESSURE_QUEUE_SIZE.set(self._current_count)
        except Exception:
            pass

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

    async def dispatch(self, request: Request, call_next) -> Response:
        # Check if backpressure is enabled
        """Execute the dispatch routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        if not settings.BACKPRESSURE_ENABLED:
            return await call_next(request)

        # Bypass for health checks and metrics
        if request.url.path in self.BYPASS_PATHS:
            return await call_next(request)

        backpressure = get_backpressure()

        # Try to acquire a processing slot
        acquired = await backpressure.acquire()

        if not acquired:
            # System at capacity
            BACKPRESSURE_REJECTED.inc()
            logger.warning(
                f"[Backpressure] Rejecting request: at capacity "
                f"({backpressure.current_load}/{backpressure.max_concurrent})"
            )
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
            await backpressure.release()
