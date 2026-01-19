# -*- coding: utf-8 -*-
"""
redis_client.py — Redis Client with Connection Pooling
-------------------------------------------------------
Provides a thread-safe Redis client with:
- Connection pooling for better performance
- Automatic reconnection with exponential backoff
- Health check support
- Graceful degradation when Redis is unavailable
"""

import os
import redis
import logging
import time
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))
REDIS_SOCKET_CONNECT_TIMEOUT = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5.0"))

# ==============================================================================
# Connection Pool (Singleton)
# ==============================================================================

_redis_pool: Optional[redis.ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None
_pool_initialized = False


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
    global _redis_pool, _redis_client, _pool_initialized

    # Return cached client if available and healthy
    if _redis_client is not None:
        try:
            _redis_client.ping()
            return _redis_client
        except Exception:
            # Connection lost, will retry below
            _redis_client = None

    # Initialize pool if needed
    if not _pool_initialized:
        _redis_pool = _create_pool()
        _pool_initialized = True

    if _redis_pool is None:
        return None

    # Try to connect with exponential backoff
    deadline = time.time() + max_wait_s
    retry_delay = 0.1

    while time.time() < deadline:
        try:
            _redis_client = redis.Redis(connection_pool=_redis_pool)
            _redis_client.ping()
            logger.info(f"[redis_client] Connected successfully")
            return _redis_client

        except redis.ConnectionError as e:
            logger.warning(f"[redis_client] Connection error: {e}")
        except redis.TimeoutError as e:
            logger.warning(f"[redis_client] Timeout error: {e}")
        except Exception as e:
            logger.warning(f"[redis_client] Unexpected error: {e}")

        time.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 1.0)  # Exponential backoff, max 1s

    logger.error(
        f"[redis_client] Failed to connect after {max_wait_s}s "
        f"({REDIS_HOST}:{REDIS_PORT})"
    )
    return None


def get_redis_async_safe() -> Optional[redis.Redis]:
    """
    Get Redis client without blocking (for use in async contexts).
    Returns None immediately if not connected.
    """
    global _redis_client

    if _redis_client is not None:
        try:
            # Use a very short timeout for the ping
            _redis_client.ping()
            return _redis_client
        except Exception:
            return None

    return None


# ==============================================================================
# Health Check
# ==============================================================================

def check_redis_health() -> dict:
    """
    Check Redis health and return detailed status.

    Returns:
        Dict with health status, latency, and pool info
    """
    result = {
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
                result["connections_in_use"] = len(_redis_pool._in_use_connections)
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
    global _redis_pool, _redis_client, _pool_initialized

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
    logger.info("[redis_client] Connections closed")
