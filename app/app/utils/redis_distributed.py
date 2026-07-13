# Objective: Redis-backed distributed coordination for multi-replica production.
"""Global semaphores, idempotency, and tenant rate limits via Redis."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional

from ..settings_dynamic import settings
from .redis_async_ops import redis_get_raw, redis_pipeline_execute, redis_set_str

logger = logging.getLogger(__name__)


def _redis_required() -> bool:
    env = str(settings.get("ENV", "development") or "development").lower()
    flag = str(settings.get("REDIS_REQUIRED_IN_PRODUCTION", "1")).strip().lower()
    return env == "production" and flag in {"1", "true", "yes", "on"}


async def redis_sliding_window_limit(
    key: str,
    *,
    max_requests: int,
    window_seconds: int,
) -> bool:
    """Return True when the key is over its sliding-window quota."""
    now = time.time()
    cutoff = now - window_seconds
    redis_key = f"tenant-rl:{key}"

    def _build(pipe):
        pipe.zremrangebyscore(redis_key, 0, cutoff)
        pipe.zcard(redis_key)
        pipe.zadd(redis_key, {str(now): now})
        pipe.expire(redis_key, window_seconds + 5)

    try:
        results = await redis_pipeline_execute(_build)
        if results is None:
            if _redis_required():
                raise RuntimeError("Redis indisponível (obrigatório em produção).")
            return False
        count = int(results[1] or 0)
        return count >= max_requests
    except Exception as exc:
        logger.warning("[redis_distributed] rate limit error: %s", exc)
        if _redis_required():
            raise
        return False


class RedisGlobalSemaphore:
    """Distributed concurrency cap using Redis INCR/DECR."""

    PREFIX = "bp:sem:"

    def __init__(self, name: str, limit: int) -> None:
        self._name = name
        self._limit = max(1, int(limit))
        self._acquired = False

    async def acquire(self) -> bool:
        from .redis_async_ops import redis_pipeline_execute as _pipe

        key = f"{self.PREFIX}{self._name}"

        def _build(pipe):
            pipe.incr(key)
            pipe.expire(key, 120)

        try:
            results = await _pipe(_build)
            if results is None:
                return True
            current = int(results[0] or 0)
            if current > self._limit:
                def _dec(pipe):
                    pipe.decr(key)

                await _pipe(_dec)
                return False
            self._acquired = True
            return True
        except Exception as exc:
            logger.warning("[redis_distributed] semaphore acquire failed: %s", exc)
            return True

    async def release(self) -> None:
        if not self._acquired:
            return
        key = f"{self.PREFIX}{self._name}"

        def _build(pipe):
            pipe.decr(key)

        try:
            await redis_pipeline_execute(_build)
        except Exception:
            pass
        self._acquired = False


async def redis_idempotency_get(key: str) -> Optional[dict[str, Any]]:
    try:
        raw = await redis_get_raw(f"idemp:{key}")
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        return json.loads(raw)
    except Exception:
        return None


async def redis_idempotency_set(key: str, payload: dict[str, Any], ttl_s: int = 300) -> None:
    await redis_set_str(f"idemp:{key}", json.dumps(payload, default=str), ttl_s=max(30, ttl_s))


def compute_idempotency_key(
    *,
    tenant_id: Optional[str],
    query: str,
    modality: str,
    model: str = "",
) -> str:
    raw = f"{tenant_id or 'anon'}|{modality}|{model}|{query}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
