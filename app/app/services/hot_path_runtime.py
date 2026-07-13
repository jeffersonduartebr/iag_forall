# Objective: Async/offloaded helpers for synchronous hot-path governance and routing.
"""Non-blocking adapters for sync helpers still used on the request path."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from ..guardrails import GuardrailDecision, check_input_guardrails
from ..roadmap_features import BudgetCheck, check_tenant_budget, get_active_policy, record_tenant_usage
from ..router_strategy import choose_top2_models
from ..utils.redis_async_ops import redis_get_str, redis_set_str

logger = logging.getLogger(__name__)

_ACTIVE_POLICY_REDIS_KEY = "hotpath:active_policy:v1"
_TENANT_BUDGET_REDIS_PREFIX = "hotpath:tenant_budget:v1:"
_REDIS_CACHE_TTL_S = 30


async def _redis_get_json(key: str) -> Optional[Any]:
    raw = await redis_get_str(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


async def _redis_set_json(key: str, value: Any, *, ttl_s: int = _REDIS_CACHE_TTL_S) -> None:
    await redis_set_str(key, json.dumps(value, default=str), ttl_s=ttl_s)


async def check_input_guardrails_async(prompt: str) -> GuardrailDecision:
    """Run guardrail checks off the event loop."""
    return await asyncio.to_thread(check_input_guardrails, prompt)


async def check_tenant_budget_async(tenant_id: Optional[str], projected_cost_usd: float = 0.0) -> BudgetCheck:
    """Run tenant budget validation off the event loop."""
    return await asyncio.to_thread(check_tenant_budget, tenant_id, projected_cost_usd)


async def get_active_policy_async() -> Optional[Dict[str, Any]]:
    """Resolve the active policy using Redis cache and thread-offloaded DB reads."""
    cached = await _redis_get_json(_ACTIVE_POLICY_REDIS_KEY)
    if isinstance(cached, dict):
        return cached or None

    policy = await asyncio.to_thread(get_active_policy)
    await _redis_set_json(_ACTIVE_POLICY_REDIS_KEY, policy or {}, ttl_s=_REDIS_CACHE_TTL_S)
    return policy


async def choose_top2_models_async(
    *,
    candidates: list[str],
    weights: Dict[str, float],
    query_text: str,
    modality: str = "text",
    uncertainty_score: float = 0.0,
    min_quality: float = 0.0,
) -> list[str]:
    """Run NSGA strategy scoring off the event loop."""
    return await asyncio.to_thread(
        choose_top2_models,
        candidates,
        weights,
        query_text,
        modality,
        uncertainty_score,
        min_quality,
    )


def schedule_tenant_usage(
    *,
    tenant_id: Optional[str],
    cost_usd: float,
    tokens_in: int = 0,
    tokens_out: int = 0,
    requests: int = 1,
) -> None:
    """Persist tenant usage without blocking the response path."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        record_tenant_usage(
            tenant_id,
            cost_usd=cost_usd,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            requests=requests,
        )
        return

    loop.create_task(
        asyncio.to_thread(
            record_tenant_usage,
            tenant_id,
            cost_usd=cost_usd,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            requests=requests,
        )
    )


async def invalidate_active_policy_cache_async() -> None:
    """Clear cross-worker active-policy cache after admin updates."""
    from app.utils.redis_async_ops import redis_delete

    await redis_delete(_ACTIVE_POLICY_REDIS_KEY)
