# -*- coding: utf-8 -*-
# Objective: Utility helpers for pricing.
"""
pricing.py - Model Cost Calculator with Redis Caching (Quick Win #4)
---------------------------------------------------------------------
Provides model cost calculations with a two-tier cache:
1. Redis cache (TTL 1 hour) - shared across workers
2. In-memory cache (TTL 5 min) - process-local fallback
"""
import time
import json
import logging
from sqlalchemy import text
from app.db import get_engine
from app.utils.redis_client import get_redis_async_safe, ensure_redis_connected

logger = logging.getLogger("pricing")

# In-memory L1 cache
_PRICING_CACHE = {}
_LAST_UPDATE = 0
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
        # Defaults (fallback for unconfigured models)
        m_lower = model.lower()
        if "gpt-5" in m_lower and "mini" in m_lower: pricing = {"in": 0.00025, "out": 0.002}
        elif "gpt-5" in m_lower: pricing = {"in": 0.00125, "out": 0.01}
        elif "gpt-4.1-mini" in m_lower: pricing = {"in": 0.0004, "out": 0.0016}
        elif "gpt-4o-mini" in m_lower: pricing = {"in": 0.00015, "out": 0.0006}
        elif "gpt-4o" in m_lower: pricing = {"in": 0.0025, "out": 0.01}

        elif "gemini-2.5-flash" in m_lower: pricing = {"in": 0.0003, "out": 0.0025}
        elif "gemini-2.5" in m_lower: pricing = {"in": 0.00125, "out": 0.01}

        elif "haiku" in m_lower: pricing = {"in": 0.001, "out": 0.005}
        elif "sonnet" in m_lower: pricing = {"in": 0.003, "out": 0.015}
        elif "opus" in m_lower: pricing = {"in": 0.005, "out": 0.025}

        else: pricing = {"in": 0.0, "out": 0.0}  # Ollama/Local

    cost_in = (input_tokens / 1000) * pricing["in"]
    cost_out = (output_tokens / 1000) * pricing["out"]

    return cost_in + cost_out
