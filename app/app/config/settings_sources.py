# Objective: Configuration support code for settings sources.
"""Source resolution helpers for dynamic settings."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def decode_redis_value(value: Any) -> Optional[str]:
    """Normalize Redis return values to strings."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return None


def resolve_setting_value(
    *,
    key: str,
    fallback: Any,
    defaults: Dict[str, str],
    cache_get: Callable[[str], Optional[Any]],
    cache_set: Callable[[str, Any], None],
    redis_get: Callable[[str], Optional[str]],
    db_get: Callable[[str], Optional[str]],
    env_get: Callable[[str], Optional[str]],
) -> Any:
    """Resolve a setting through cache, Redis, DB, env, then defaults."""
    cached = cache_get(key)
    if cached is not None:
        return cached

    value = redis_get(key)
    if value is None:
        value = db_get(key)
    if value is None:
        value = env_get(key)
    if value is None:
        value = defaults.get(key, fallback)

    if value is not None:
        cache_set(key, value)
    return value
