"""Thin runtime-facing adapter over roadmap/governance helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..roadmap_features import check_tenant_budget, ensure_roadmap_tables, get_active_policy, record_tenant_usage


def ensure_runtime_support_tables() -> None:
    """Ensure optional governance tables used by runtime hooks exist."""
    return ensure_roadmap_tables()


def get_runtime_active_policy() -> Optional[Dict[str, Any]]:
    """Return the currently active governance policy for runtime selection."""
    return get_active_policy()


def check_runtime_budget(tenant_id: Optional[str]):
    """Validate tenant budget before running the query path."""
    return check_tenant_budget(tenant_id)


def record_runtime_usage(*, tenant_id: Optional[str], cost_usd: float, tokens_in: int, tokens_out: int, requests: int) -> None:
    """Record request cost and token usage for runtime accounting."""
    record_tenant_usage(
        tenant_id=tenant_id,
        cost_usd=cost_usd,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        requests=requests,
    )
