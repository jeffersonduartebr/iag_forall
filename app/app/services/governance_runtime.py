# Objective: Service-layer helpers for governance runtime.
"""Thin runtime-facing adapter over roadmap/governance helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..roadmap_features import check_tenant_budget, ensure_roadmap_tables, get_active_policy, record_tenant_usage
from .hot_path_runtime import (
    check_tenant_budget_async,
    get_active_policy_async,
    invalidate_active_policy_cache_async,
    schedule_tenant_usage,
)


def _auto_ddl_enabled() -> bool:
    from ..settings_dynamic import settings

    env = str(settings.get("ENV", "development") or "development").lower()
    flag = settings.get("ROADMAP_AUTO_DDL")
    if flag is None or str(flag).strip() == "":
        return env not in {"production", "prod"}
    return str(flag).strip().lower() in {"1", "true", "yes", "on"}


def ensure_runtime_support_tables() -> None:
    """Ensure optional governance tables used by runtime hooks exist."""
    if not _auto_ddl_enabled():
        return None
    return ensure_roadmap_tables()


def get_runtime_active_policy() -> Optional[Dict[str, Any]]:
    """Return the currently active governance policy for runtime selection."""
    return get_active_policy()


async def get_runtime_active_policy_async() -> Optional[Dict[str, Any]]:
    """Return the active policy without blocking the event loop."""
    return await get_active_policy_async()


def check_runtime_budget(tenant_id: Optional[str]):
    """Validate tenant budget before running the query path."""
    return check_tenant_budget(tenant_id)


async def check_runtime_budget_async(tenant_id: Optional[str]):
    """Validate tenant budget without blocking the event loop."""
    return await check_tenant_budget_async(tenant_id)


def record_runtime_usage(*, tenant_id: Optional[str], cost_usd: float, tokens_in: int, tokens_out: int, requests: int) -> None:
    """Record request cost and token usage for runtime accounting."""
    record_tenant_usage(
        tenant_id=tenant_id,
        cost_usd=cost_usd,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        requests=requests,
    )


def schedule_runtime_usage(*, tenant_id: Optional[str], cost_usd: float, tokens_in: int, tokens_out: int, requests: int) -> None:
    """Queue tenant usage persistence without blocking the response path."""
    schedule_tenant_usage(
        tenant_id=tenant_id,
        cost_usd=cost_usd,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        requests=requests,
    )


async def invalidate_runtime_policy_cache_async() -> None:
    """Invalidate cross-worker active-policy cache after admin updates."""
    await invalidate_active_policy_cache_async()
