# Objective: Async Redis helpers for request hot paths.
"""Non-blocking Redis operations built on ``redis.asyncio``."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from app.config.settings_sources import decode_redis_value
from app.utils.redis_client import get_redis_async


async def redis_get_raw(key: str) -> Optional[Any]:
    """Fetch one Redis key using the async client."""
    client = await get_redis_async()
    if client is None:
        return None
    try:
        return await client.get(key)
    except Exception:
        return None


async def redis_get_str(key: str) -> Optional[str]:
    """Fetch and decode one Redis string key."""
    return decode_redis_value(await redis_get_raw(key))


async def redis_set_str(key: str, value: str, *, ttl_s: Optional[int] = None) -> bool:
    """Set one Redis string key, optionally with TTL."""
    client = await get_redis_async()
    if client is None:
        return False
    try:
        if ttl_s is None:
            await client.set(key, value)
        else:
            await client.set(key, value, ex=max(1, int(ttl_s)))
        return True
    except Exception:
        return False


async def redis_delete(key: str) -> None:
    """Delete one Redis key."""
    client = await get_redis_async()
    if client is None:
        return
    try:
        await client.delete(key)
    except Exception:
        return


async def redis_hgetall_map(key: str) -> Dict[str, str]:
    """Return a hash as ``{field: value}`` strings."""
    client = await get_redis_async()
    if client is None:
        return {}
    try:
        raw = await client.hgetall(key)
        if not raw:
            return {}
        out: Dict[str, str] = {}
        for field, value in raw.items():
            field_s = field.decode() if isinstance(field, bytes) else str(field)
            out[field_s] = decode_redis_value(value) or ""
        return out
    except Exception:
        return {}


async def redis_pipeline_execute(build: Callable[[Any], None]) -> Optional[List[Any]]:
    """Execute one async Redis pipeline built by ``build(pipe)``."""
    client = await get_redis_async()
    if client is None:
        return None
    try:
        pipe = client.pipeline()
        build(pipe)
        return await pipe.execute()
    except Exception:
        return None
