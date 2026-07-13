# Objective: Configuration support code for settings sources.
"""Source resolution helpers for dynamic settings."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

# Secrets and bootstrap keys are env-only (never overridden by Redis/DB).
SECRET_SETTING_KEYS = frozenset({
    "ADMIN_TOKEN",
    "ADMIN_TOKEN_PREVIOUS",
    "API_KEYS",
    "JWT_SECRET",
    "REDIS_PASSWORD",
    "DB_PASS",
    "MYSQL_ROOT_PASSWORD",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "METRICS_TOKEN",
    "REQUIRE_API_AUTH",
    "ENV",
})


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
    """Resolve a setting through cache, sources, then defaults.

    Precedence:
    - ``FORCE_<KEY>`` env overrides everything for emergency operations.
    - Secret/bootstrap keys: env -> defaults (never Redis/DB).
    - Runtime tunables: Redis -> DB -> env -> defaults.
    """
    cached = cache_get(key)
    if cached is not None:
        return cached

    force_key = f"FORCE_{key}"
    force_value = env_get(force_key)
    if force_value is not None:
        cache_set(key, force_value)
        return force_value

    if key in SECRET_SETTING_KEYS:
        value = env_get(key)
        if value is None:
            value = defaults.get(key, fallback)
        if value is not None:
            cache_set(key, value)
        return value

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


async def resolve_setting_value_async(
    *,
    key: str,
    fallback: Any,
    defaults: Dict[str, str],
    cache_get: Callable[[str], Optional[Any]],
    cache_set: Callable[[str, Any], None],
    redis_get_async: Callable[[str], Any],
    db_get: Callable[[str], Optional[str]],
    env_get: Callable[[str], Optional[str]],
) -> Any:
    """Async variant of :func:`resolve_setting_value` for request hot paths."""
    cached = cache_get(key)
    if cached is not None:
        return cached

    force_key = f"FORCE_{key}"
    force_value = env_get(force_key)
    if force_value is not None:
        cache_set(key, force_value)
        return force_value

    if key in SECRET_SETTING_KEYS:
        value = env_get(key)
        if value is None:
            value = defaults.get(key, fallback)
        if value is not None:
            cache_set(key, value)
        return value

    value = await redis_get_async(key)
    if value is None:
        value = db_get(key)
    if value is None:
        value = env_get(key)
    if value is None:
        value = defaults.get(key, fallback)

    if value is not None:
        cache_set(key, value)
    return value
