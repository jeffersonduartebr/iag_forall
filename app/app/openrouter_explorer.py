# Objective: Explore OpenRouter catalog models outside the fixed candidate pool.
"""OpenRouter exploration — discover cloud models via sampled traffic."""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.openrouter_catalog import fetch_openrouter_models
from app.openrouter_exploration_policy import (  # noqa: F401  (reexportados)
    ExplorationConfig,
    _blended_usd_per_1k,
    _catalog_usd_per_1k_for_model,
    _cost_tier,
    _effective_exploration_rate,
    _enrich_models_with_cost_comparison,
    _failure_rate,
    _model_slug,
    _observed_usd_per_1k,
    _parse_allowlist,
    _pick_ucb_from_pool,
    _price_within_budget,
    _provider_allowed,
    _running_mean,
    _setting,
    evaluate_promotion,
    load_exploration_config,
    next_exploration_stats,
)
from app.openrouter_exploration_state import (  # noqa: F401  (reexportados; chamadores resolvem por este módulo)
    REDIS_BLOCKLIST_KEY,
    REDIS_DAILY_COUNTER_KEY,
    REDIS_DAILY_DATE_KEY,
    REDIS_DAILY_USD_KEY,
    REDIS_MODEL_STATS_PREFIX,
    REDIS_MODELS_SET_KEY,
    _get_blocklist,
    _get_daily_count,
    _get_daily_usd,
    _increment_daily,
    _load_many_model_stats,
    _load_model_stats,
    _save_blocklist,
    _save_model_stats,
    _track_model_key,
)
from app.openrouter_shadow import maybe_run_shadow_comparison  # noqa: F401  (reexportado)

logger = logging.getLogger(__name__)

REDIS_SUGGESTIONS_KEY = "openrouter:explore:suggestions"
REDIS_POOL_CACHE_KEY = "openrouter:explore:pool_cache"
REDIS_AUTO_PROMOTED_KEY = "openrouter:explore:auto_promoted"










def _openrouter_configured() -> bool:
    from app.openrouter_catalog import get_openrouter_api_key

    return bool(get_openrouter_api_key())


























async def _get_redis():
    # O cliente *async*: todo chamador aguarda. ``get_redis_async_safe`` (alias obsoleto) devolve o cliente
    # SÍNCRONO apesar do nome; aguardar seu set/get/incr falhava em silêncio (produção, 2026-09-25): stats de
    # exploração nunca salvas e os tetos diários de contagem/USD nunca contados.
    from app.utils.redis_client import get_redis_async

    return await get_redis_async()






async def blocklist_model(model: str) -> None:
    """Add a model to the local exploration blocklist."""
    rds = await _get_redis()
    if not rds:
        return
    blocklist = await _get_blocklist(rds)
    blocklist.add(model)
    await _save_blocklist(rds, blocklist)
    await invalidate_exploration_pool_cache()


async def unblocklist_model(model: str) -> None:
    """Remove a model from the exploration blocklist."""
    rds = await _get_redis()
    if not rds:
        return
    blocklist = await _get_blocklist(rds)
    blocklist.discard(model)
    await _save_blocklist(rds, blocklist)
    await invalidate_exploration_pool_cache()


async def invalidate_exploration_pool_cache() -> None:
    rds = await _get_redis()
    if not rds:
        return
    try:
        await rds.delete(REDIS_POOL_CACHE_KEY)
    except Exception:
        pass














def _persist_stats_to_db(model: str, stats: Dict[str, Any], *, auto_promoted: bool = False) -> None:
    try:
        from sqlalchemy import text

        from app.db import get_engine

        catalog = stats.get("catalog_usd_per_1k") or {}
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO openrouter_exploration_stats (
                        model, count, failure_count, mean_reward, mean_latency_s, mean_cost_usd,
                        mean_observed_usd_per_1k, catalog_prompt_usd_per_1k, catalog_completion_usd_per_1k,
                        auto_promoted_at, blocklisted, stats_json
                    ) VALUES (
                        :model, :count, :failure_count, :mean_reward, :mean_latency_s, :mean_cost_usd,
                        :mean_observed_usd_per_1k, :catalog_prompt_usd_per_1k, :catalog_completion_usd_per_1k,
                        CASE WHEN :auto_promoted = 1 THEN CURRENT_TIMESTAMP ELSE NULL END,
                        0, :stats_json
                    )
                    ON DUPLICATE KEY UPDATE
                        count = VALUES(count),
                        failure_count = VALUES(failure_count),
                        mean_reward = VALUES(mean_reward),
                        mean_latency_s = VALUES(mean_latency_s),
                        mean_cost_usd = VALUES(mean_cost_usd),
                        mean_observed_usd_per_1k = VALUES(mean_observed_usd_per_1k),
                        catalog_prompt_usd_per_1k = VALUES(catalog_prompt_usd_per_1k),
                        catalog_completion_usd_per_1k = VALUES(catalog_completion_usd_per_1k),
                        auto_promoted_at = COALESCE(VALUES(auto_promoted_at), auto_promoted_at),
                        stats_json = VALUES(stats_json),
                        updated_at = CURRENT_TIMESTAMP
                    """
                ),
                {
                    "model": model,
                    "count": int(stats.get("count", 0)),
                    "failure_count": int(stats.get("failure_count", 0)),
                    "mean_reward": stats.get("mean_reward"),
                    "mean_latency_s": stats.get("mean_latency_s"),
                    "mean_cost_usd": stats.get("mean_cost_usd"),
                    "mean_observed_usd_per_1k": stats.get("mean_observed_usd_per_1k"),
                    "catalog_prompt_usd_per_1k": catalog.get("prompt_usd_per_1k"),
                    "catalog_completion_usd_per_1k": catalog.get("completion_usd_per_1k"),
                    "auto_promoted": 1 if auto_promoted else 0,
                    "stats_json": json.dumps(stats),
                },
            )
    except Exception as exc:
        logger.debug("[openrouter_explore] DB persist skipped for %s: %s", model, exc)


async def auto_promote_to_candidates(model: str, settings: Any) -> bool:
    """Automatically append an explored model to CANDIDATE_MODELS_LIST."""
    cfg = load_exploration_config(settings)
    if not cfg.auto_promote_enabled:
        return False
    if not model.startswith("openrouter/"):
        return False

    getter = getattr(settings, "CANDIDATE_MODELS_LIST", None)
    if callable(getter):
        candidates = list(getter() or [])
    else:
        candidates = list(getattr(settings, "CANDIDATE_MODELS_LIST", []) or [])

    if model in candidates:
        return False

    candidates.append(model)
    settings.set("CANDIDATE_MODELS_LIST", candidates, actor="openrouter_explorer", source="auto_promote")
    await invalidate_exploration_pool_cache()

    rds = await _get_redis()
    if rds:
        try:
            raw = await rds.get(REDIS_AUTO_PROMOTED_KEY)
            promoted = json.loads(raw) if raw else []
            if not isinstance(promoted, list):
                promoted = []
            promoted.append({"model": model, "promoted_at": time.time()})
            await rds.set(REDIS_AUTO_PROMOTED_KEY, json.dumps(promoted[-100:]))
        except Exception:
            pass

    try:
        from app.observability import OPENROUTER_EXPLORATION_OUTCOMES

        OPENROUTER_EXPLORATION_OUTCOMES.labels(model=model, promotable="auto_promoted").inc()
    except Exception:
        pass

    logger.info("[openrouter_explore] auto-promoted %s to CANDIDATE_MODELS_LIST", model)
    return True


async def _cached_pool(rds, known_models: Set[str], blocklist: Set[str]) -> Optional[List[str]]:
    """Pool cached in Redis while not expired (minus known/blocked models)."""
    if not rds:
        return None
    try:
        cached = await rds.get(REDIS_POOL_CACHE_KEY)
        if not cached:
            return None
        payload = json.loads(cached.decode() if isinstance(cached, bytes) else str(cached))
        pool = payload.get("pool") if isinstance(payload, dict) and payload.get("expires_at", 0) > time.time() else None
    except Exception:
        return None
    return [m for m in pool if m not in known_models and m not in blocklist] if isinstance(pool, list) else None


def _eligible_catalog_models(catalog: List[Dict[str, Any]], known_models: Set[str], blocklist: Set[str], cfg) -> List[str]:
    """OpenRouter catalog entries not yet known/blocked, allowed provider and within the price budget."""
    eligible: List[str] = []
    for item in catalog:
        full_name = str(item.get("full_name") or "")
        if not full_name.startswith("openrouter/") or full_name in known_models or full_name in blocklist:
            continue
        slug = _model_slug(full_name)
        # ":free" são endpoints gratuitos: limitados por taxa (429 em série no teste de 2026-09-25) e, em geral,
        # com retenção/treino sobre os prompts — incompatível com dados de estudantes. Nunca são explorados.
        if ":free" not in slug and _provider_allowed(slug, cfg) and _price_within_budget(slug, cfg):
            eligible.append(full_name)
    return eligible


def _explore_score(full_name: str, stats: Dict[str, Any], cfg) -> Optional[float]:
    """Exploration priority (fewer observations first; cheaper first in cost_hunt); None = skip."""
    if int(stats.get("consecutive_failures", 0)) >= cfg.consecutive_failure_block:
        return None
    if stats.get("auto_promoted_at") and evaluate_promotion(stats, cfg)["promotable"]:
        return None
    score = 1.0 / (1.0 + int(stats.get("count", 0)))
    if cfg.mode == "cost_hunt":
        catalog_cost = _catalog_usd_per_1k_for_model(full_name)
        if catalog_cost:
            blended = _blended_usd_per_1k(catalog_cost["prompt_usd_per_1k"], catalog_cost["completion_usd_per_1k"])
            score *= 1.0 / (1.0 + blended * 1000.0)
    return score


async def _build_exploration_pool(
    known_models: Set[str],
    cfg: ExplorationConfig,
    modality: str,
) -> List[str]:
    """Top ``pool_size`` catalog models to explore (text only), cached in Redis for ``pool_cache_ttl_s``."""
    if modality != "text":
        return []
    rds = await _get_redis()
    blocklist = await _get_blocklist(rds)
    cached = await _cached_pool(rds, known_models, blocklist)
    if cached is not None:
        return cached

    catalog = await fetch_openrouter_models()
    if not catalog:
        return []
    known_openrouter = {m for m in known_models if m.startswith("openrouter/")}
    eligible = _eligible_catalog_models(catalog, known_openrouter, blocklist, cfg)
    stats_by_model = await _load_many_model_stats(rds, eligible)
    scored = [(name, _explore_score(name, stats_by_model.get(name, {}), cfg)) for name in eligible]
    candidates = sorted(((n, s) for n, s in scored if s is not None), key=lambda x: x[1], reverse=True)
    if not candidates:
        return []
    pool = [name for name, _ in candidates[: cfg.pool_size]]
    if rds:
        try:
            await rds.set(
                REDIS_POOL_CACHE_KEY,
                json.dumps({"pool": pool, "expires_at": time.time() + cfg.pool_cache_ttl_s}),
            )
        except Exception:
            pass
    return pool


async def maybe_pick_exploration_model(
    *,
    known_models: Set[str],
    modality: str,
    settings: Any,
    uncertainty_score: float = 0.5,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return an exploration model and debug metadata, or None to use normal routing."""
    cfg = load_exploration_config(settings)
    if not cfg.enabled or not _openrouter_configured():
        return None
    if modality != "text":
        return None

    effective_rate = _effective_exploration_rate(cfg, uncertainty_score)
    if random.random() >= effective_rate:
        return None

    rds = await _get_redis()
    daily = await _get_daily_count(rds)
    if daily >= cfg.max_per_day:
        logger.info("[openrouter_explore] daily cap reached (%s)", cfg.max_per_day)
        return None

    daily_usd = await _get_daily_usd(rds)
    if cfg.max_usd_per_day > 0 and daily_usd >= cfg.max_usd_per_day:
        logger.info("[openrouter_explore] daily USD cap reached (%.4f)", cfg.max_usd_per_day)
        return None

    pool = await _build_exploration_pool(known_models, cfg, modality)
    if not pool:
        return None

    stats_map = {m: await _load_model_stats(rds, m) for m in pool} if rds else {}
    chosen = _pick_ucb_from_pool(pool, stats_map)

    shadow_compare = random.random() < cfg.shadow_compare_rate

    return chosen, {
        "openrouter_exploration": True,
        "exploration_pool_size": len(pool),
        "exploration_rate": effective_rate,
        "exploration_mode": cfg.mode,
        "shadow_compare": shadow_compare,
        "adaptive_rate": cfg.adaptive_rate_enabled,
        "uncertainty_score": uncertainty_score,
    }


async def _record_suggestion(rds, model: str, updated: Dict[str, Any], promo: Dict[str, Any]) -> None:
    """Add a promotable model to the admin suggestion list (last 50, no duplicates)."""
    if not rds:
        return
    try:
        raw = await rds.get(REDIS_SUGGESTIONS_KEY)
        suggestions = json.loads(raw) if raw else []
        if not isinstance(suggestions, list):
            suggestions = []
        if any(s.get("model") == model for s in suggestions if isinstance(s, dict)):
            return
        suggestions.append(
            {
                "model": model,
                "mean_reward": updated["mean_reward"],
                "mean_latency_s": updated.get("mean_latency_s"),
                "mean_observed_usd_per_1k": updated.get("mean_observed_usd_per_1k"),
                "catalog_usd_per_1k": updated.get("catalog_usd_per_1k"),
                "count": updated["count"],
                "suggested_at": time.time(),
                "promotion_passed": promo.get("promotion_passed"),
            }
        )
        await rds.set(REDIS_SUGGESTIONS_KEY, json.dumps(suggestions[-50:]))
    except Exception as exc:
        logger.warning("[openrouter_explore] suggestion update failed: %s", exc)


async def _auto_promote(rds, model: str, updated: Dict[str, Any], settings: Any) -> bool:
    """Promote to the candidate list and persist the stats (flagged when promoted)."""
    if not await auto_promote_to_candidates(model, settings):
        _persist_stats_to_db(model, updated)
        return False
    updated["auto_promoted_at"] = time.time()
    await _save_model_stats(rds, model, updated)
    _persist_stats_to_db(model, updated, auto_promoted=True)
    return True


async def record_exploration_outcome(
    *,
    model: str,
    reward: float,
    latency_s: float,
    cost_usd: float,
    settings: Any,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    success: bool = True,
    judge_quality: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Persist exploration stats, auto-promote when eligible, return promotion info."""
    if not model.startswith("openrouter/"):
        return None

    rds = await _get_redis()
    cfg = load_exploration_config(settings)
    stats = await _load_model_stats(rds, model) if rds else {}
    updated = next_exploration_stats(
        stats,
        reward=reward,
        latency_s=latency_s,
        cost_usd=cost_usd,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        success=success,
        judge_quality=judge_quality,
        catalog_usd_per_1k=stats.get("catalog_usd_per_1k") or _catalog_usd_per_1k_for_model(model),
    )
    promo = evaluate_promotion(updated, cfg)
    updated.update(promo)

    await _save_model_stats(rds, model, updated)
    await _increment_daily(rds, cost_usd if success else 0.0)

    promotion_result: Dict[str, Any] = {"auto_promoted": False, "model": model}
    if promo["promotable"] and success:
        await _record_suggestion(rds, model, updated, promo)
        if cfg.auto_promote_enabled and not updated.get("auto_promoted_at"):
            promotion_result["auto_promoted"] = await _auto_promote(rds, model, updated, settings)
        else:
            _persist_stats_to_db(model, updated)
    else:
        _persist_stats_to_db(model, updated)

    promotion_result["promotable"] = promo["promotable"]
    promotion_result["promotion_blockers"] = promo.get("promotion_blockers", [])

    try:
        from app.observability import OPENROUTER_EXPLORATION_OUTCOMES

        OPENROUTER_EXPLORATION_OUTCOMES.labels(model=model, promotable=str(promo["promotable"])).inc()
    except Exception:
        pass

    return promotion_result


async def record_exploration_failure(*, model: str, settings: Any, error: str = "") -> None:
    """Record a failed exploration attempt without full outcome metrics."""
    await record_exploration_outcome(
        model=model,
        reward=0.0,
        latency_s=0.0,
        cost_usd=0.0,
        settings=settings,
        success=False,
    )
    if error:
        logger.info("[openrouter_explore] failure recorded for %s: %s", model, error)




async def get_exploration_status(settings: Any) -> Dict[str, Any]:
    """Return exploration config, daily usage, model stats, and promotion suggestions."""
    cfg = load_exploration_config(settings)
    rds = await _get_redis()
    daily = await _get_daily_count(rds) if rds else 0
    daily_usd = await _get_daily_usd(rds) if rds else 0.0

    models: List[Dict[str, Any]] = []
    suggestions: List[Dict[str, Any]] = []
    auto_promoted: List[Dict[str, Any]] = []
    blocklist: List[str] = []

    if rds:
        try:
            raw_suggestions = await rds.get(REDIS_SUGGESTIONS_KEY)
            if raw_suggestions:
                parsed = json.loads(raw_suggestions)
                if isinstance(parsed, list):
                    suggestions = parsed
        except Exception:
            suggestions = []

        try:
            raw_promoted = await rds.get(REDIS_AUTO_PROMOTED_KEY)
            if raw_promoted:
                parsed = json.loads(raw_promoted)
                if isinstance(parsed, list):
                    auto_promoted = parsed
        except Exception:
            auto_promoted = []

        blocklist = sorted(await _get_blocklist(rds))

        try:
            model_keys = await rds.smembers(REDIS_MODELS_SET_KEY)
            for key in model_keys or []:
                key_str = key.decode() if isinstance(key, bytes) else str(key)
                stats = await _load_model_stats(rds, key_str)
                if not stats:
                    continue
                if not stats.get("catalog_usd_per_1k"):
                    catalog = _catalog_usd_per_1k_for_model(key_str)
                    if catalog:
                        stats["catalog_usd_per_1k"] = {
                            "prompt_usd_per_1k": round(float(catalog["prompt_usd_per_1k"]), 8),
                            "completion_usd_per_1k": round(float(catalog["completion_usd_per_1k"]), 8),
                        }
                promo = evaluate_promotion(stats, cfg)
                models.append(
                    {
                        "model": key_str,
                        **stats,
                        **promo,
                        "blocklisted": key_str in blocklist,
                    }
                )
        except Exception as exc:
            logger.warning("[openrouter_explore] status scan failed: %s", exc)

    models.sort(key=lambda m: (-float(m.get("mean_reward", 0)), -int(m.get("count", 0))))
    cost_benchmark = _enrich_models_with_cost_comparison(models)

    return {
        "enabled": cfg.enabled and _openrouter_configured(),
        "setting_enabled": cfg.enabled,
        "openrouter_configured": _openrouter_configured(),
        "config": {
            "mode": cfg.mode,
            "rate": cfg.rate,
            "max_per_day": cfg.max_per_day,
            "max_usd_per_day": cfg.max_usd_per_day,
            "max_price_prompt_per_1k": cfg.max_price_prompt_per_1k,
            "max_price_completion_per_1k": cfg.max_price_completion_per_1k,
            "promote_min_samples": cfg.promote_min_samples,
            "promote_min_reward": cfg.promote_min_reward,
            "promote_max_latency_s": cfg.promote_max_latency_s,
            "promote_max_cost_usd_per_1k": cfg.promote_max_cost_usd_per_1k,
            "promote_max_failure_rate": cfg.promote_max_failure_rate,
            "provider_allowlist": list(cfg.provider_allowlist),
            "pool_size": cfg.pool_size,
            "pool_cache_ttl_s": cfg.pool_cache_ttl_s,
            "adaptive_rate_enabled": cfg.adaptive_rate_enabled,
            "auto_promote_enabled": cfg.auto_promote_enabled,
            "shadow_compare_rate": cfg.shadow_compare_rate,
            "consecutive_failure_block": cfg.consecutive_failure_block,
        },
        "usage": {
            "explorations_today": daily,
            "remaining_today": max(0, cfg.max_per_day - daily),
            "usd_spent_today": round(daily_usd, 6),
            "usd_remaining_today": round(max(0.0, cfg.max_usd_per_day - daily_usd), 6) if cfg.max_usd_per_day > 0 else None,
        },
        "cost_benchmark": cost_benchmark,
        "models": models,
        "suggestions": suggestions,
        "auto_promoted": auto_promoted,
        "blocklist": blocklist,
    }
