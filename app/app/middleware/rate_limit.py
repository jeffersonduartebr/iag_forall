# -*- coding: utf-8 -*-
"""
rate_limit.py — Rate Limiting Middleware with Redis Support
------------------------------------------------------------
Provides rate limiting functionality with support for:
- Redis-based distributed rate limiting (recommended for multiple workers)
- In-memory fallback when Redis is unavailable
- Sliding window algorithm for accurate rate limiting

Usage:
    from app.middleware.rate_limit import RateLimitMiddleware, rate_limit_store

    app.add_middleware(RateLimitMiddleware)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config.constants import (
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW,
    RATE_LIMIT_CLEANUP_INTERVAL,
    RATE_LIMIT_ENTRY_TTL,
)

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
_redis_client = None


def _get_redis():
    """Lazy Redis client getter."""
    global _redis_client
    try:
        from app.utils.redis_client import get_redis_async_safe, ensure_redis_connected
        _redis_client = get_redis_async_safe() or ensure_redis_connected(max_wait_s=0.0, min_retry_interval_s=2.0)
    except Exception:
        _redis_client = None
    return _redis_client


class RateLimitStore:
    """
    Rate limiting store with Redis support and in-memory fallback.

    For production with multiple workers, uses Redis for distributed rate limiting.
    Falls back to in-memory store when Redis is unavailable.
    """

    # Redis key prefix for rate limit data
    REDIS_PREFIX = "ratelimit:"
    REDIS_TTL = 3600  # 1 hour TTL for Redis keys

    def __init__(self):
        self._memory_store: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
        self._use_redis: Optional[bool] = None
        self._next_redis_probe_at = 0.0
        self._redis_reprobe_interval_s = 10.0

    def _should_use_redis(self) -> bool:
        """Check if Redis is available and should be used."""
        now = time.time()
        should_probe = self._use_redis is None or (self._use_redis is False and now >= self._next_redis_probe_at)
        if should_probe:
            rds = _get_redis()
            self._use_redis = rds is not None
            if self._use_redis:
                self._next_redis_probe_at = 0.0
                logger.info("[rate_limit] Using Redis-based rate limiting")
            else:
                self._next_redis_probe_at = now + self._redis_reprobe_interval_s
                logger.warning("[rate_limit] Redis unavailable, using in-memory rate limiting")
        return self._use_redis

    async def is_rate_limited(
        self,
        client_ip: str,
        max_requests: int = RATE_LIMIT_REQUESTS,
        window_seconds: int = RATE_LIMIT_WINDOW,
    ) -> bool:
        """
        Check if client is rate limited using sliding window algorithm.

        Args:
            client_ip: Client IP address
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds

        Returns:
            True if rate limited, False otherwise
        """
        if self._should_use_redis():
            return await self._is_rate_limited_redis(client_ip, max_requests, window_seconds)
        return await self._is_rate_limited_memory(client_ip, max_requests, window_seconds)

    async def _is_rate_limited_redis(
        self, client_ip: str, max_requests: int, window_seconds: int
    ) -> bool:
        """Redis-based rate limiting using sorted sets."""
        rds = _get_redis()
        if rds is None:
            # Redis connection lost, fallback to memory
            self._use_redis = False
            self._next_redis_probe_at = time.time() + self._redis_reprobe_interval_s
            return await self._is_rate_limited_memory(client_ip, max_requests, window_seconds)

        try:
            key = f"{self.REDIS_PREFIX}{client_ip}"
            now = time.time()
            cutoff = now - window_seconds

            # Use pipeline for atomic operations
            pipe = rds.pipeline()
            # Remove old entries
            pipe.zremrangebyscore(key, 0, cutoff)
            # Count current entries
            pipe.zcard(key)
            # Add current request
            pipe.zadd(key, {str(now): now})
            # Set TTL to auto-cleanup
            pipe.expire(key, self.REDIS_TTL)
            results = pipe.execute()

            count = results[1]  # zcard result

            if count >= max_requests:
                return True

            return False

        except Exception as e:
            logger.warning(f"[rate_limit] Redis error, falling back to memory: {e}")
            self._use_redis = False
            self._next_redis_probe_at = time.time() + self._redis_reprobe_interval_s
            return await self._is_rate_limited_memory(client_ip, max_requests, window_seconds)

    async def _is_rate_limited_memory(
        self, client_ip: str, max_requests: int, window_seconds: int
    ) -> bool:
        """In-memory rate limiting with sliding window."""
        now = time.time()
        cutoff = now - window_seconds

        async with self._lock:
            if client_ip not in self._memory_store:
                self._memory_store[client_ip] = []

            # Clean old entries
            self._memory_store[client_ip] = [
                t for t in self._memory_store[client_ip] if t > cutoff
            ]

            # Check if over limit
            if len(self._memory_store[client_ip]) >= max_requests:
                return True

            # Add current request
            self._memory_store[client_ip].append(now)
            return False

    async def cleanup(self) -> int:
        """
        Remove old entries to prevent memory bloat.

        Returns:
            Number of entries cleaned up
        """
        now = time.time()
        cutoff = now - RATE_LIMIT_ENTRY_TTL
        cleaned = 0

        async with self._lock:
            for ip in list(self._memory_store.keys()):
                before_len = len(self._memory_store[ip])
                self._memory_store[ip] = [t for t in self._memory_store[ip] if t > cutoff]
                cleaned += before_len - len(self._memory_store[ip])

                if not self._memory_store[ip]:
                    del self._memory_store[ip]

        return cleaned

    def get_stats(self) -> dict:
        """Get rate limit store statistics."""
        return {
            "using_redis": self._use_redis,
            "memory_entries": len(self._memory_store),
            "memory_total_requests": sum(
                len(v) for v in self._memory_store.values()
            ),
        }


# Global rate limit store instance
rate_limit_store = RateLimitStore()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.

    Supports both Redis-based and in-memory rate limiting.
    """

    # Paths exempted from rate limiting
    EXEMPT_PATHS = frozenset([
        "/health",
        "/healthz",
        "/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    ])

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and metrics
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Get client IP (support for X-Forwarded-For)
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        # Check rate limit
        if await rate_limit_store.is_rate_limited(
            client_ip, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW
        ):
            # Record metric
            try:
                from app.observability import RATE_LIMIT_EXCEEDED
                RATE_LIMIT_EXCEEDED.labels(client_ip=client_ip).inc()
            except Exception:
                pass

            logger.warning(
                "rate_limit_exceeded",
                extra={
                    "client_ip": client_ip,
                    "limit": RATE_LIMIT_REQUESTS,
                    "window": RATE_LIMIT_WINDOW,
                },
            )

            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "message": "Rate limit exceeded. Please try again later.",
                    "retry_after": RATE_LIMIT_WINDOW,
                },
                headers={
                    "Retry-After": str(RATE_LIMIT_WINDOW),
                    "X-RateLimit-Limit": str(RATE_LIMIT_REQUESTS),
                    "X-RateLimit-Window": str(RATE_LIMIT_WINDOW),
                },
            )

        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Window"] = str(RATE_LIMIT_WINDOW)

        return response


async def periodic_cleanup():
    """Periodically clean up rate limit store to prevent memory bloat."""
    while True:
        await asyncio.sleep(RATE_LIMIT_CLEANUP_INTERVAL)
        try:
            cleaned = await rate_limit_store.cleanup()
            if cleaned > 0:
                logger.debug(f"[rate_limit] Cleaned up {cleaned} old entries")
        except Exception as e:
            logger.warning(f"[rate_limit] Cleanup failed: {e}")
