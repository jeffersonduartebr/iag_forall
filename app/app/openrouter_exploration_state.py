# -*- coding: utf-8 -*-
# Objective: Redis state helpers for OpenRouter exploration (stats, daily budget, blocklist).
"""Redis-backed exploration state. Every helper receives the async Redis client.

Extracted from ``app.openrouter_explorer`` (re-exported there); the callers
stay in the explorer so they resolve these names through its namespace.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Set

logger = logging.getLogger("app.openrouter_explorer")

REDIS_MODEL_STATS_PREFIX = "openrouter:explore:model:"
REDIS_MODELS_SET_KEY = "openrouter:explore:models"
REDIS_DAILY_COUNTER_KEY = "openrouter:explore:daily_count"
REDIS_DAILY_DATE_KEY = "openrouter:explore:daily_date"
REDIS_DAILY_USD_KEY = "openrouter:explore:daily_usd"
REDIS_BLOCKLIST_KEY = "openrouter:explore:blocklist"


async def _get_blocklist(rds) -> Set[str]:
    if not rds:
        return set()
    try:
        raw = await rds.get(REDIS_BLOCKLIST_KEY)
        if not raw:
            return set()
        text = raw.decode() if isinstance(raw, bytes) else str(raw)
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return {str(x) for x in parsed if x}
    except Exception:
        pass
    return set()


async def _save_blocklist(rds, models: Set[str]) -> None:
    if not rds:
        return
    try:
        await rds.set(REDIS_BLOCKLIST_KEY, json.dumps(sorted(models)))
    except Exception as exc:
        logger.warning("[openrouter_explore] blocklist save failed: %s", exc)


async def _track_model_key(rds, full_name: str) -> None:
    if not rds:
        return
    try:
        await rds.sadd(REDIS_MODELS_SET_KEY, full_name)
    except Exception:
        pass


async def _get_daily_count(rds) -> int:
    if not rds:
        return 0
    try:
        today = time.strftime("%Y-%m-%d")
        stored_date = await rds.get(REDIS_DAILY_DATE_KEY)
        date_str = stored_date.decode() if isinstance(stored_date, bytes) else str(stored_date or "")
        if date_str != today:
            await rds.set(REDIS_DAILY_DATE_KEY, today)
            await rds.set(REDIS_DAILY_COUNTER_KEY, 0)
            await rds.set(REDIS_DAILY_USD_KEY, "0")
            return 0
        raw = await rds.get(REDIS_DAILY_COUNTER_KEY)
        return int(raw or 0)
    except Exception as exc:
        logger.debug("[openrouter_explore] daily count read failed: %s", exc)
        return 0


async def _get_daily_usd(rds) -> float:
    if not rds:
        return 0.0
    try:
        today = time.strftime("%Y-%m-%d")
        stored_date = await rds.get(REDIS_DAILY_DATE_KEY)
        date_str = stored_date.decode() if isinstance(stored_date, bytes) else str(stored_date or "")
        if date_str != today:
            return 0.0
        raw = await rds.get(REDIS_DAILY_USD_KEY)
        return float(raw or 0.0)
    except Exception:
        return 0.0


async def _increment_daily(rds, cost_usd: float = 0.0) -> None:
    if not rds:
        return
    try:
        # Vira o dia antes de somar: sem isto, um outcome logo após a meia-noite herdava o contador de ontem.
        await _get_daily_count(rds)
        await rds.incr(REDIS_DAILY_COUNTER_KEY)
        if cost_usd > 0:
            await rds.incrbyfloat(REDIS_DAILY_USD_KEY, float(cost_usd))
        await rds.expire(REDIS_DAILY_COUNTER_KEY, 86400 * 2)
        await rds.expire(REDIS_DAILY_USD_KEY, 86400 * 2)
    except Exception as exc:
        logger.debug("[openrouter_explore] daily increment failed: %s", exc)


def _decode_stats(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


async def _load_model_stats(rds, full_name: str) -> Dict[str, Any]:
    if not rds:
        return {}
    try:
        return _decode_stats(await rds.get(f"{REDIS_MODEL_STATS_PREFIX}{full_name}"))
    except Exception:
        return {}


async def _load_many_model_stats(rds, full_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Stats for many models in one MGET (falls back to one GET per model)."""
    if not rds or not full_names:
        return {}
    try:
        raws = await rds.mget([f"{REDIS_MODEL_STATS_PREFIX}{name}" for name in full_names])
        return {name: _decode_stats(raw) for name, raw in zip(full_names, raws)}
    except Exception:
        return {name: await _load_model_stats(rds, name) for name in full_names}


async def _save_model_stats(rds, full_name: str, stats: Dict[str, Any]) -> None:
    if not rds:
        return
    try:
        await rds.set(f"{REDIS_MODEL_STATS_PREFIX}{full_name}", json.dumps(stats))
        await _track_model_key(rds, full_name)
    except Exception as exc:
        logger.warning("[openrouter_explore] stats save failed for %s: %s", full_name, exc)
