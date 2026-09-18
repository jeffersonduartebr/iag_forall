# -*- coding: utf-8 -*-
# Objective: Utility helpers for pricing.
"""
pricing.py - Model Cost Calculator with Redis Caching (Quick Win #4)
---------------------------------------------------------------------
Provides model cost calculations with a two-tier cache:
1. Redis cache (TTL 1 hour) - shared across workers
2. In-memory cache (TTL 5 min) - process-local fallback
"""
import json
import logging
import math
import time
from typing import Any

from sqlalchemy import text

from app.db import get_engine
from app.utils.redis_client import ensure_redis_connected, get_redis_async_safe

logger = logging.getLogger("pricing")

# In-memory L1 cache
_PRICING_CACHE: dict[str, Any] = {}
_LAST_UPDATE = 0.0
CACHE_TTL = 300  # 5 minutes for local cache

# Redis L2 cache keys
REDIS_PRICING_KEY = "pricing:model_costs"
REDIS_PRICING_TTL = 3600  # 1 hour for Redis cache

def _get_rds():
    # Non-blocking on hot path; opportunistic reconnect.
    """Execute the get rds routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    return get_redis_async_safe() or ensure_redis_connected(max_wait_s=0.0, min_retry_interval_s=2.0)

# Metrics imports (safe import to avoid circular deps)
try:
    from app.observability import PRICING_CACHE_HITS, PRICING_CACHE_MISSES
except ImportError:
    PRICING_CACHE_HITS = None
    PRICING_CACHE_MISSES = None


def _refresh_pricing_from_db() -> dict:
    """Load pricing data from database."""
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(text("SELECT model, cost_input_1k, cost_output_1k FROM model_pricing")).fetchall()
            new_cache = {}
            for r in rows:
                new_cache[r[0]] = {"in": float(r[1]), "out": float(r[2])}
            return new_cache
    except Exception as e:
        logger.warning(f"[pricing] DB refresh error: {e}")
        return {}


def _refresh_pricing():
    """Refresh pricing cache from Redis or DB."""
    global _PRICING_CACHE, _LAST_UPDATE

    # Try Redis first (Quick Win #4)
    rds = _get_rds()
    if rds:
        try:
            cached = rds.get(REDIS_PRICING_KEY)
            if cached:
                _PRICING_CACHE = json.loads(cached)
                _LAST_UPDATE = time.time()
                if PRICING_CACHE_HITS:
                    PRICING_CACHE_HITS.inc()
                return
        except Exception as e:
            logger.debug(f"[pricing] Redis read error: {e}")

    # Fallback to DB
    if PRICING_CACHE_MISSES:
        PRICING_CACHE_MISSES.inc()

    new_cache = _refresh_pricing_from_db()
    if new_cache:
        _PRICING_CACHE = new_cache
        _LAST_UPDATE = time.time()

        # Store in Redis for other workers
        rds = _get_rds()
        if rds:
            try:
                rds.setex(REDIS_PRICING_KEY, REDIS_PRICING_TTL, json.dumps(new_cache))
            except Exception as e:
                logger.debug(f"[pricing] Redis write error: {e}")


def invalidate_pricing_cache():
    """Invalidate pricing cache (call on settings hot-reload)."""
    global _PRICING_CACHE, _LAST_UPDATE
    _PRICING_CACHE = {}
    _LAST_UPDATE = 0

    rds = _get_rds()
    if rds:
        try:
            rds.delete(REDIS_PRICING_KEY)
        except Exception:
            pass

    logger.info("[pricing] Cache invalidated")


def get_model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate total cost in USD."""
    if time.time() - _LAST_UPDATE > CACHE_TTL:
        _refresh_pricing()

    # Try exact match or clean model name
    pricing = _PRICING_CACHE.get(model)
    if not pricing:
        clean_model = model.split("/", 1)[1] if "/" in model else model
        pricing = _PRICING_CACHE.get(clean_model)

    if not pricing:
        slug_candidate = model.split("/", 1)[1] if model.startswith("openrouter/") else clean_model
        if "/" in slug_candidate:
            try:
                from app.openrouter_catalog import get_openrouter_pricing_per_1k

                pricing = get_openrouter_pricing_per_1k(slug_candidate)
            except Exception:
                pricing = None

    if not pricing:
        # Defaults (fallback for unconfigured models)
        # Preços de lista por 1k tokens (OpenRouter, 2026-09-18) para modelos sem
        # linha em model_pricing. Ordem importa: variantes específicas antes das genéricas.
        m_lower = model.lower()
        if "gpt-5.5" in m_lower and "pro" in m_lower: pricing = {"in": 0.030, "out": 0.180}
        elif "gpt-5.5" in m_lower: pricing = {"in": 0.005, "out": 0.030}
        elif "gpt-5" in m_lower and "mini" in m_lower: pricing = {"in": 0.00025, "out": 0.002}
        elif "gpt-5" in m_lower: pricing = {"in": 0.00125, "out": 0.01}
        elif "gpt-4.1-mini" in m_lower: pricing = {"in": 0.0004, "out": 0.0016}
        elif "gpt-4o-mini" in m_lower: pricing = {"in": 0.00015, "out": 0.0006}
        elif "gpt-4o" in m_lower: pricing = {"in": 0.0025, "out": 0.01}

        elif "gemini-2.5-flash" in m_lower: pricing = {"in": 0.0003, "out": 0.0025}
        elif "gemini-2.5" in m_lower: pricing = {"in": 0.00125, "out": 0.01}

        elif "fable" in m_lower: pricing = {"in": 0.010, "out": 0.050}
        elif "haiku" in m_lower: pricing = {"in": 0.001, "out": 0.005}
        elif "sonnet-5" in m_lower: pricing = {"in": 0.002, "out": 0.010}
        elif "sonnet" in m_lower: pricing = {"in": 0.003, "out": 0.015}
        elif "opus" in m_lower: pricing = {"in": 0.005, "out": 0.025}

        else: pricing = {"in": 0.0, "out": 0.0}  # Ollama/Local

    cost_in = (input_tokens / 1000) * pricing["in"]
    cost_out = (output_tokens / 1000) * pricing["out"]

    return cost_in + cost_out


# ============================================================
# Custo imputado da inferência local
# ============================================================
# Modelos locais não geram cobrança por token (get_model_cost devolve 0), mas
# ocupam o equipamento. O custo de inferência imputa esse tempo de ocupação:
#
#   taxa_USD/h = LOCAL_COST_USD_PER_HOUR, quando > 0; senão
#              = P_hw / (vida_anos * 8760 * utilização) + (W / 1000) * tarifa_USD/kWh
#   C_local    = (t_ocupação_s / 3600) * taxa_USD/h / slots_paralelos
#
# O valor imputado nunca entra no custo de caixa (orçamento/cobrança por tenant).


def _local_cost_setting(key: str, default: float) -> float:
    try:
        from app.settings_dynamic import settings

        value = float(settings.get(key, default))
        return value if math.isfinite(value) else default
    except Exception:
        return default


def is_local_model(model: str) -> bool:
    """Return whether ``model`` runs on the local Ollama server."""
    return "ollama" in str(model or "").lower()


def local_cost_rate_usd_per_hour() -> float:
    """Hourly cost of the local inference host (amortization + energy)."""
    override = _local_cost_setting("LOCAL_COST_USD_PER_HOUR", 0.0)
    if override > 0:
        return override
    price = max(0.0, _local_cost_setting("LOCAL_COST_HW_PRICE_USD", 0.0))
    years = _local_cost_setting("LOCAL_COST_HW_LIFETIME_YEARS", 3.0)
    utilization = _local_cost_setting("LOCAL_COST_HW_UTILIZATION", 0.5)
    power_w = max(0.0, _local_cost_setting("LOCAL_COST_POWER_W", 0.0))
    tariff = max(0.0, _local_cost_setting("LOCAL_COST_ENERGY_USD_PER_KWH", 0.0))
    productive_hours = years * 8760.0 * utilization
    amortization = price / productive_hours if productive_hours > 0 else 0.0
    return amortization + (power_w / 1000.0) * tariff


def impute_local_cost(occupancy_s: float) -> float:
    """Imputed USD cost of occupying the local host for ``occupancy_s`` seconds."""
    if _local_cost_setting("LOCAL_COST_IMPUTATION_ENABLED", 1.0) <= 0:
        return 0.0
    try:
        seconds = float(occupancy_s)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(seconds) or seconds <= 0:
        return 0.0
    slots = max(1.0, _local_cost_setting("LOCAL_COST_PARALLEL_SLOTS", 1.0))
    return seconds / 3600.0 * local_cost_rate_usd_per_hour() / slots
