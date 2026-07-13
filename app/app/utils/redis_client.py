# -*- coding: utf-8 -*-
# Objective: Utility helpers for redis client.
"""
redis_client.py — Redis Client with Connection Pooling
-------------------------------------------------------
Provides a thread-safe Redis client with:
- Connection pooling for better performance
- Automatic reconnection with exponential backoff
- Health check support
- Graceful degradation when Redis is unavailable
"""

import logging
import os
import time
from contextlib import contextmanager
from threading import Lock
from typing import Any, Optional

import redis

logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))  # Optimized for high-capacity
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))
REDIS_SOCKET_CONNECT_TIMEOUT = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5.0"))

# ==============================================================================
# Connection Pool (Singleton)
# ==============================================================================

_redis_pool: Optional[redis.ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None
_pool_initialized = False
_connect_lock = Lock()
_last_connect_attempt_s = 0.0


def _create_pool() -> Optional[redis.ConnectionPool]:
    """Create a Redis connection pool."""
    try:
        pool = redis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            max_connections=REDIS_MAX_CONNECTIONS,
            socket_timeout=REDIS_SOCKET_TIMEOUT,
            socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT,
            decode_responses=False,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        logger.info(
            f"[redis_client] Connection pool created: "
            f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB} "
            f"(max_connections={REDIS_MAX_CONNECTIONS}, auth={'yes' if REDIS_PASSWORD else 'no'})"
        )
        return pool
    except Exception as e:
        logger.error(f"[redis_client] Failed to create connection pool: {e}")
        return None


def _ensure_pool_initialized() -> Optional[redis.ConnectionPool]:
    """Initialize Redis pool once and return it."""
    global _redis_pool, _pool_initialized
    if not _pool_initialized:
        _redis_pool = _create_pool()
        _pool_initialized = True
    return _redis_pool


def _connect_once() -> Optional[redis.Redis]:
    """Attempt a single Redis connection without sleeping."""
    global _redis_client
    pool = _ensure_pool_initialized()
    if pool is None:
        return None
    try:
        _redis_client = redis.Redis(connection_pool=pool)
        _redis_client.ping()
        return _redis_client
    except Exception:
        _redis_client = None
        return None


def ensure_redis_connected(
    max_wait_s: float = 0.0,
    min_retry_interval_s: float = 1.0,
) -> Optional[redis.Redis]:
    """
    Ensure Redis connection is available.

    - max_wait_s <= 0: non-blocking single attempt.
    - max_wait_s > 0: bounded retry loop with backoff.
    """
    global _redis_client, _last_connect_attempt_s

    if _redis_client is not None:
        try:
            _redis_client.ping()
            return _redis_client
        except Exception:
            _redis_client = None

    now = time.monotonic()
    if min_retry_interval_s > 0 and (now - _last_connect_attempt_s) < min_retry_interval_s:
        return None

    with _connect_lock:
        now = time.monotonic()
        if min_retry_interval_s > 0 and (now - _last_connect_attempt_s) < min_retry_interval_s:
            return None
        _last_connect_attempt_s = now

        if max_wait_s <= 0:
            return _connect_once()

        deadline = time.monotonic() + max_wait_s
        retry_delay = 0.1
        while time.monotonic() < deadline:
            client = _connect_once()
            if client is not None:
                logger.info("[redis_client] Connected successfully")
                return client
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 1.0)

    logger.error(
        f"[redis_client] Failed to connect after {max_wait_s}s "
        f"({REDIS_HOST}:{REDIS_PORT})"
    )
    return None


def get_redis(max_wait_s: int = 5) -> Optional[redis.Redis]:
    """
    Get a Redis client from the connection pool.

    Features:
    - Uses connection pooling for better performance
    - Automatic retry with exponential backoff
    - Returns None if Redis is unavailable (graceful degradation)

    Args:
        max_wait_s: Maximum time to wait for connection

    Returns:
        Redis client or None if unavailable
    """
    return ensure_redis_connected(max_wait_s=float(max_wait_s), min_retry_interval_s=0.0)


def get_redis_async_safe() -> Optional[redis.Redis]:
    """
    Get Redis client without blocking (for use in async contexts).
    Returns None immediately if not connected.

    Deprecated name: prefer ``get_redis_sync_nonblocking()``.
    """
    return get_redis_sync_nonblocking()


def get_redis_sync_nonblocking() -> Optional[redis.Redis]:
    """Non-blocking sync Redis client for legacy code paths."""
    return ensure_redis_connected(max_wait_s=0.0, min_retry_interval_s=1.0)


# ==============================================================================
# Async Redis Client (hot path)
# ==============================================================================

_async_redis_client = None
_async_redis_lock = Lock()


async def get_redis_async():
    """Return a shared ``redis.asyncio`` client for async request handlers."""
    global _async_redis_client
    if _async_redis_client is not None:
        try:
            await _async_redis_client.ping()
            return _async_redis_client
        except Exception:
            _async_redis_client = None

    with _async_redis_lock:
        if _async_redis_client is not None:
            return _async_redis_client
        try:
            import redis.asyncio as aioredis

            _async_redis_client = aioredis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=False,
                socket_timeout=REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT,
            )
            await _async_redis_client.ping()
            return _async_redis_client
        except Exception as exc:
            logger.warning("[redis_client] Async Redis unavailable: %s", exc)
            return None


async def close_redis_async() -> None:
    """Close the async Redis client."""
    global _async_redis_client
    if _async_redis_client is not None:
        try:
            await _async_redis_client.close()
        except Exception:
            pass
        _async_redis_client = None


# ==============================================================================
# Health Check
# ==============================================================================

def check_redis_health() -> dict:
    """
    Check Redis health and return detailed status.

    Returns:
        Dict with health status, latency, and pool info
    """
    result: dict[str, Any] = {
        "healthy": False,
        "latency_ms": None,
        "pool_size": REDIS_MAX_CONNECTIONS,
        "connections_in_use": None,
        "error": None,
    }

    client = get_redis_async_safe()
    if client is None:
        result["error"] = "Not connected"
        return result

    try:
        start = time.time()
        client.ping()
        latency_ms = (time.time() - start) * 1000

        # Get pool stats if available
        if _redis_pool is not None:
            try:
                result["connections_in_use"] = len(_redis_pool._in_use_connections)  # type: ignore[attr-defined]  # private runtime attr
            except AttributeError:
                pass  # Pool implementation may vary

        result["healthy"] = True
        result["latency_ms"] = round(latency_ms, 2)

    except Exception as e:
        result["error"] = str(e)

    return result


# ==============================================================================
# Utility Functions
# ==============================================================================

@contextmanager
def redis_pipeline():
    """
    Context manager for Redis pipeline operations.

    Usage:
        with redis_pipeline() as pipe:
            pipe.set("key1", "value1")
            pipe.set("key2", "value2")
            results = pipe.execute()
    """
    client = get_redis()
    if client is None:
        raise RuntimeError("Redis not available")

    pipe = client.pipeline()
    try:
        yield pipe
        pipe.execute()
    except Exception:
        pipe.reset()
        raise


def close_redis():
    """Close Redis connections and pool. Call on application shutdown."""
    global _redis_pool, _redis_client, _pool_initialized, _last_connect_attempt_s

    if _redis_client is not None:
        try:
            _redis_client.close()
        except Exception:
            pass
        _redis_client = None

    if _redis_pool is not None:
        try:
            _redis_pool.disconnect()
        except Exception:
            pass
        _redis_pool = None

    _pool_initialized = False
    _last_connect_attempt_s = 0.0
    logger.info("[redis_client] Connections closed")
