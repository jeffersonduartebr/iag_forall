# -*- coding: utf-8 -*-
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
        """Resumo do comportamento desta função.

        Returns:
            Valor retornado pela função.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Resumo do comportamento desta função.

        Returns:
            Valor retornado pela função.
        """
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
        # Non-blocking check
        if self._semaphore.locked() and self._current_count >= self._max_concurrent:
            return False

        try:
            # Try to acquire without blocking
            acquired = self._semaphore.locked()
            if not acquired:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=0.001)
                async with self._count_lock:
                    self._current_count += 1
                    BACKPRESSURE_QUEUE_SIZE.set(self._current_count)
                return True
            return False
        except asyncio.TimeoutError:
            return False
        except Exception:
            return True  # On error, allow request through

    async def release(self):
        """Release a processing slot."""
        try:
            self._semaphore.release()
            async with self._count_lock:
                self._current_count = max(0, self._current_count - 1)
                BACKPRESSURE_QUEUE_SIZE.set(self._current_count)
        except ValueError:
            # Semaphore released too many times
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
        """Resumo do comportamento desta função.

        Args:
            request: Parâmetro de entrada.
            call_next: Parâmetro de entrada.

        Returns:
            Valor retornado pela função.
        """
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
