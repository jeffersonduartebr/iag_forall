# Objective: Explore OpenRouter catalog models outside the fixed candidate pool.
"""OpenRouter exploration — discover cloud models via sampled traffic."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from app.openrouter_catalog import fetch_openrouter_models, get_openrouter_pricing_per_1k

logger = logging.getLogger(__name__)

REDIS_MODEL_STATS_PREFIX = "openrouter:explore:model:"
REDIS_MODELS_SET_KEY = "openrouter:explore:models"
REDIS_DAILY_COUNTER_KEY = "openrouter:explore:daily_count"
REDIS_DAILY_DATE_KEY = "openrouter:explore:daily_date"
REDIS_DAILY_USD_KEY = "openrouter:explore:daily_usd"
REDIS_SUGGESTIONS_KEY = "openrouter:explore:suggestions"
REDIS_POOL_CACHE_KEY = "openrouter:explore:pool_cache"
REDIS_BLOCKLIST_KEY = "openrouter:explore:blocklist"
REDIS_AUTO_PROMOTED_KEY = "openrouter:explore:auto_promoted"


@dataclass
class ExplorationConfig:
    """Runtime configuration for OpenRouter catalog exploration."""

    enabled: bool
    mode: str
    rate: float
    max_per_day: int
    max_usd_per_day: float
    max_price_prompt_per_1k: float
    max_price_completion_per_1k: float
    promote_min_samples: int
    promote_min_reward: float
    promote_max_latency_s: float
    promote_max_cost_usd_per_1k: float
    promote_max_failure_rate: float
    provider_allowlist: Tuple[str, ...]
    pool_size: int
    pool_cache_ttl_s: int
    adaptive_rate_enabled: bool
    auto_promote_enabled: bool
    shadow_compare_rate: float
    consecutive_failure_block: int
    promote_eval_gate_enabled: bool
    promote_min_judge_quality: float


def _setting(settings: Any, key: str, default: Any) -> Any:
    getter = getattr(settings, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(settings, key, default)


def _parse_allowlist(raw: Any) -> Tuple[str, ...]:
    if isinstance(raw, (list, tuple)):
        return tuple(str(x).strip() for x in raw if str(x).strip())
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return tuple(str(x).strip() for x in parsed if str(x).strip())
            except json.JSONDecodeError:
                pass
        return tuple(p.strip() for p in text.split(",") if p.strip())
    return ()


def load_exploration_config(settings: Any) -> ExplorationConfig:
    """Build exploration config from dynamic settings."""
    try:
        from app.services.frozen_policy import frozen_exploration_enabled, frozen_exploration_rate

        enabled = frozen_exploration_enabled(
            str(_setting(settings, "OPENROUTER_EXPLORATION_ENABLED", "0")).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        rate = frozen_exploration_rate(max(0.0, min(1.0, float(_setting(settings, "OPENROUTER_EXPLORATION_RATE", "0.10")))))
    except Exception:
        enabled = str(_setting(settings, "OPENROUTER_EXPLORATION_ENABLED", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        rate = max(0.0, min(1.0, float(_setting(settings, "OPENROUTER_EXPLORATION_RATE", "0.10"))))
    allowlist = _parse_allowlist(_setting(settings, "OPENROUTER_EXPLORATION_PROVIDER_ALLOWLIST", "[]"))
    auto_promote = str(_setting(settings, "OPENROUTER_EXPLORATION_AUTO_PROMOTE_ENABLED", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    adaptive = str(_setting(settings, "OPENROUTER_EXPLORATION_ADAPTIVE_RATE_ENABLED", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return ExplorationConfig(
        enabled=enabled,
        mode=str(_setting(settings, "OPENROUTER_EXPLORATION_MODE", "balanced")).strip().lower() or "balanced",
        rate=rate,
        max_per_day=max(1, int(_setting(settings, "OPENROUTER_EXPLORATION_MAX_PER_DAY", "100"))),
        max_usd_per_day=max(0.0, float(_setting(settings, "OPENROUTER_EXPLORATION_MAX_USD_PER_DAY", "3.0"))),
        max_price_prompt_per_1k=float(_setting(settings, "OPENROUTER_EXPLORATION_MAX_PRICE_PROMPT_1K", "0.01")),
        max_price_completion_per_1k=float(_setting(settings, "OPENROUTER_EXPLORATION_MAX_PRICE_COMPLETION_1K", "0.03")),
        promote_min_samples=max(1, int(_setting(settings, "OPENROUTER_EXPLORATION_PROMOTE_MIN_SAMPLES", "15"))),
        promote_min_reward=max(0.0, min(1.0, float(_setting(settings, "OPENROUTER_EXPLORATION_PROMOTE_MIN_REWARD", "0.72")))),
        promote_max_latency_s=max(1.0, float(_setting(settings, "OPENROUTER_EXPLORATION_PROMOTE_MAX_LATENCY_S", "30.0"))),
        promote_max_cost_usd_per_1k=max(0.0, float(_setting(settings, "OPENROUTER_EXPLORATION_PROMOTE_MAX_COST_USD_PER_1K", "0.02"))),
        promote_max_failure_rate=max(0.0, min(1.0, float(_setting(settings, "OPENROUTER_EXPLORATION_PROMOTE_MAX_FAILURE_RATE", "0.20")))),
        provider_allowlist=allowlist,
        pool_size=max(5, int(_setting(settings, "OPENROUTER_EXPLORATION_POOL_SIZE", "40"))),
        pool_cache_ttl_s=max(60, int(_setting(settings, "OPENROUTER_EXPLORATION_POOL_CACHE_TTL_S", "600"))),
        adaptive_rate_enabled=adaptive,
        auto_promote_enabled=auto_promote,
        shadow_compare_rate=max(0.0, min(1.0, float(_setting(settings, "OPENROUTER_EXPLORATION_SHADOW_COMPARE_RATE", "0.05")))),
        consecutive_failure_block=max(1, int(_setting(settings, "OPENROUTER_EXPLORATION_CONSECUTIVE_FAILURE_BLOCK", "3"))),
        promote_eval_gate_enabled=str(_setting(settings, "OPENROUTER_EXPLORATION_PROMOTE_EVAL_GATE_ENABLED", "1")).strip().lower()
        in {"1", "true", "yes", "on"},
        promote_min_judge_quality=max(0.0, float(_setting(settings, "OPENROUTER_EXPLORATION_PROMOTE_MIN_JUDGE_QUALITY", "6.5"))),
    )


def _openrouter_configured() -> bool:
    from app.openrouter_catalog import get_openrouter_api_key

    return bool(get_openrouter_api_key())


def _model_slug(full_name: str) -> str:
    return full_name.split("/", 1)[1] if full_name.startswith("openrouter/") else full_name


def _running_mean(prev_mean: float, count: int, value: float) -> float:
    return prev_mean + (float(value) - prev_mean) / count


def _observed_usd_per_1k(cost_usd: float, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    total_tokens = int(prompt_tokens) + int(completion_tokens)
    if total_tokens <= 0:
        return None
    return (float(cost_usd) / total_tokens) * 1000.0


def _catalog_usd_per_1k_for_model(full_name: str) -> Optional[Dict[str, float]]:
    slug = _model_slug(full_name)
    pricing = get_openrouter_pricing_per_1k(slug)
    if not pricing:
        return None
    return {
        "prompt_usd_per_1k": float(pricing["in"]),
        "completion_usd_per_1k": float(pricing["out"]),
    }


def _blended_usd_per_1k(prompt_usd: float, completion_usd: float) -> float:
    return (float(prompt_usd) + float(completion_usd)) / 2.0


def _cost_tier(observed: Optional[float], reference: Optional[float], tolerance: float = 0.15) -> str:
    if observed is None or reference is None or reference <= 0:
        return "desconhecido"
    ratio = observed / reference
    if ratio < (1.0 - tolerance):
        return "mais_barato"
    if ratio > (1.0 + tolerance):
        return "mais_caro"
    return "similar"


def _failure_rate(stats: Dict[str, Any]) -> float:
    count = int(stats.get("count", 0))
    if count <= 0:
        return 0.0
    return float(stats.get("failure_count", 0)) / count


def evaluate_promotion(stats: Dict[str, Any], cfg: ExplorationConfig) -> Dict[str, Any]:
    """Multi-objective promotion gate (reward + latency + cost + reliability)."""
    count = int(stats.get("count", 0))
    mean_reward = float(stats.get("mean_reward", 0.0))
    mean_latency = float(stats.get("mean_latency_s", 0.0))
    mean_cost_1k = stats.get("mean_observed_usd_per_1k")
    blockers: List[str] = []
    passed: List[str] = []

    if count < cfg.promote_min_samples:
        blockers.append("amostras_insuficientes")
    else:
        passed.append("amostras_ok")

    if mean_reward < cfg.promote_min_reward:
        blockers.append("reward_baixo")
    else:
        passed.append("reward_ok")

    if mean_latency > cfg.promote_max_latency_s:
        blockers.append("latencia_alta")
    else:
        passed.append("latencia_ok")

    if mean_cost_1k is not None and float(mean_cost_1k) > cfg.promote_max_cost_usd_per_1k:
        blockers.append("custo_alto")
    elif mean_cost_1k is not None:
        passed.append("custo_ok")

    if _failure_rate(stats) > cfg.promote_max_failure_rate:
        blockers.append("taxa_falha_alta")
    else:
        passed.append("confiabilidade_ok")

    if cfg.promote_eval_gate_enabled:
        mean_judge = stats.get("mean_judge_quality")
        if mean_judge is None:
            blockers.append("eval_gate_sem_judge")
        elif float(mean_judge) < cfg.promote_min_judge_quality:
            blockers.append("eval_gate_judge_baixo")
        else:
            passed.append("eval_gate_ok")

    promotable = len(blockers) == 0
    return {
        "promotable": promotable,
        "promotion_blockers": blockers,
        "promotion_passed": passed,
    }


def _enrich_models_with_cost_comparison(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    observed_values = [
        float(m["mean_observed_usd_per_1k"])
        for m in models
        if m.get("mean_observed_usd_per_1k") is not None and int(m.get("observed_cost_samples", 0)) > 0
    ]
    pool_median = 0.0
    if observed_values:
        sorted_vals = sorted(observed_values)
        mid = len(sorted_vals) // 2
        pool_median = (
            sorted_vals[mid]
            if len(sorted_vals) % 2 == 1
            else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        )

    for model in models:
        catalog = model.get("catalog_usd_per_1k") or {}
        prompt_catalog = catalog.get("prompt_usd_per_1k")
        completion_catalog = catalog.get("completion_usd_per_1k")
        if prompt_catalog is not None and completion_catalog is not None:
            model["catalog_blended_usd_per_1k"] = round(_blended_usd_per_1k(prompt_catalog, completion_catalog), 8)

        observed = model.get("mean_observed_usd_per_1k")
        reference = pool_median if pool_median > 0 else model.get("catalog_blended_usd_per_1k")
        model["cost_vs_pool_ratio"] = round(observed / reference, 4) if observed and reference else None
        model["cost_tier"] = _cost_tier(observed, reference)

    return {"pool_median_observed_usd_per_1k": round(pool_median, 8) if pool_median > 0 else None}


def _effective_exploration_rate(cfg: ExplorationConfig, uncertainty_score: float) -> float:
    if not cfg.adaptive_rate_enabled:
        return cfg.rate
    uq = max(0.0, min(1.0, float(uncertainty_score)))
    factor = 0.5 + uq
    return max(0.0, min(1.0, cfg.rate * factor))


def _price_within_budget(slug: str, cfg: ExplorationConfig) -> bool:
    pricing = get_openrouter_pricing_per_1k(slug)
    if not pricing:
        return True
    return pricing["in"] <= cfg.max_price_prompt_per_1k and pricing["out"] <= cfg.max_price_completion_per_1k


def _provider_allowed(slug: str, cfg: ExplorationConfig) -> bool:
    if not cfg.provider_allowlist:
        return True
    prefix = slug.split("/", 1)[0] if "/" in slug else slug
    return prefix in cfg.provider_allowlist


async def _get_redis():
    from app.utils.redis_client import get_redis_async_safe

    return get_redis_async_safe()


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
        today = time.strftime("%Y-%m-%d")
        await rds.set(REDIS_DAILY_DATE_KEY, today)
        await rds.incr(REDIS_DAILY_COUNTER_KEY)
        if cost_usd > 0:
            await rds.incrbyfloat(REDIS_DAILY_USD_KEY, float(cost_usd))
        await rds.expire(REDIS_DAILY_COUNTER_KEY, 86400 * 2)
        await rds.expire(REDIS_DAILY_USD_KEY, 86400 * 2)
    except Exception as exc:
        logger.debug("[openrouter_explore] daily increment failed: %s", exc)


async def _load_model_stats(rds, full_name: str) -> Dict[str, Any]:
    if not rds:
        return {}
    try:
        raw = await rds.get(f"{REDIS_MODEL_STATS_PREFIX}{full_name}")
        if not raw:
            return {}
        text = raw.decode() if isinstance(raw, bytes) else str(raw)
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


async def _save_model_stats(rds, full_name: str, stats: Dict[str, Any]) -> None:
    if not rds:
        return
    try:
        await rds.set(f"{REDIS_MODEL_STATS_PREFIX}{full_name}", json.dumps(stats))
        await _track_model_key(rds, full_name)
    except Exception as exc:
        logger.warning("[openrouter_explore] stats save failed for %s: %s", full_name, exc)


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


async def _build_exploration_pool(
    known_models: Set[str],
    cfg: ExplorationConfig,
    modality: str,
) -> List[str]:
    if modality != "text":
        return []

    rds = await _get_redis()
    blocklist = await _get_blocklist(rds)

    if rds:
        try:
            cached = await rds.get(REDIS_POOL_CACHE_KEY)
            if cached:
                text = cached.decode() if isinstance(cached, bytes) else str(cached)
                payload = json.loads(text)
                if isinstance(payload, dict) and payload.get("expires_at", 0) > time.time():
                    pool = payload.get("pool") or []
                    if isinstance(pool, list):
                        return [m for m in pool if m not in known_models and m not in blocklist]
        except Exception:
            pass

    catalog = await fetch_openrouter_models()
    if not catalog:
        return []

    known_openrouter = {m for m in known_models if m.startswith("openrouter/")}

    candidates: List[Tuple[str, float]] = []
    for item in catalog:
        full_name = str(item.get("full_name") or "")
        if not full_name.startswith("openrouter/"):
            continue
        if full_name in known_openrouter or full_name in blocklist:
            continue
        slug = _model_slug(full_name)
        if not _provider_allowed(slug, cfg):
            continue
        if not _price_within_budget(slug, cfg):
            continue

        stats = await _load_model_stats(rds, full_name) if rds else {}
        if int(stats.get("consecutive_failures", 0)) >= cfg.consecutive_failure_block:
            continue
        promo = evaluate_promotion(stats, cfg)
        if promo["promotable"] and stats.get("auto_promoted_at"):
            continue

        count = int(stats.get("count", 0))
        explore_score = 1.0 / (1.0 + count)
        if cfg.mode == "cost_hunt":
            catalog_cost = _catalog_usd_per_1k_for_model(full_name)
            if catalog_cost:
                blended = _blended_usd_per_1k(catalog_cost["prompt_usd_per_1k"], catalog_cost["completion_usd_per_1k"])
                explore_score *= 1.0 / (1.0 + blended * 1000.0)
        candidates.append((full_name, explore_score))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[1], reverse=True)
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


def _pick_ucb_from_pool(pool: List[str], stats_map: Dict[str, Dict[str, Any]]) -> str:
    total_pulls = sum(int(stats_map.get(m, {}).get("count", 0)) for m in pool) or 1
    best_model = pool[0]
    best_score = -1.0
    for model in pool:
        stats = stats_map.get(model, {})
        count = int(stats.get("count", 0))
        if count == 0:
            return model
        mean_reward = float(stats.get("mean_reward", 0.0))
        exploration_bonus = math.sqrt(2.0 * math.log(total_pulls + 1) / count)
        score = mean_reward + exploration_bonus
        if score > best_score:
            best_score = score
            best_model = model
    return best_model


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
    count = int(stats.get("count", 0)) + 1
    prev_mean_reward = float(stats.get("mean_reward", 0.0))
    prev_mean_latency = float(stats.get("mean_latency_s", 0.0))
    prev_mean_cost = float(stats.get("mean_cost_usd", 0.0))
    prev_mean_observed = float(stats.get("mean_observed_usd_per_1k", 0.0))
    observed_samples = int(stats.get("observed_cost_samples", 0))
    failure_count = int(stats.get("failure_count", 0))
    consecutive_failures = int(stats.get("consecutive_failures", 0))

    if not success:
        failure_count += 1
        consecutive_failures += 1
    else:
        consecutive_failures = 0

    mean_reward = _running_mean(prev_mean_reward, count, reward)
    mean_latency = _running_mean(prev_mean_latency, count, latency_s)
    mean_cost = _running_mean(prev_mean_cost, count, cost_usd)

    observed_usd_per_1k = _observed_usd_per_1k(cost_usd, prompt_tokens, completion_tokens)
    if observed_usd_per_1k is not None:
        observed_samples += 1
        mean_observed: Optional[float] = _running_mean(prev_mean_observed, observed_samples, observed_usd_per_1k)
    else:
        mean_observed = prev_mean_observed if observed_samples > 0 else None

    catalog_usd_per_1k = stats.get("catalog_usd_per_1k") or _catalog_usd_per_1k_for_model(model)

    updated: Dict[str, Any] = {
        "count": count,
        "failure_count": failure_count,
        "consecutive_failures": consecutive_failures,
        "failure_rate": round(_failure_rate({"count": count, "failure_count": failure_count}), 4),
        "mean_reward": round(mean_reward, 4),
        "mean_latency_s": round(mean_latency, 3),
        "mean_cost_usd": round(mean_cost, 6),
        "last_reward": round(float(reward), 4),
        "last_latency_s": round(float(latency_s), 3),
        "last_cost_usd": round(float(cost_usd), 6),
        "last_success": bool(success),
        "total_prompt_tokens": int(stats.get("total_prompt_tokens", 0)) + int(prompt_tokens),
        "total_completion_tokens": int(stats.get("total_completion_tokens", 0)) + int(completion_tokens),
        "last_prompt_tokens": int(prompt_tokens),
        "last_completion_tokens": int(completion_tokens),
        "updated_at": time.time(),
    }
    if judge_quality is not None:
        updated["last_judge_quality"] = round(float(judge_quality), 2)
        prev_judge = float(stats.get("mean_judge_quality", judge_quality))
        updated["mean_judge_quality"] = round(_running_mean(prev_judge, count, float(judge_quality)), 2)

    if catalog_usd_per_1k:
        updated["catalog_usd_per_1k"] = {
            "prompt_usd_per_1k": round(float(catalog_usd_per_1k["prompt_usd_per_1k"]), 8),
            "completion_usd_per_1k": round(float(catalog_usd_per_1k["completion_usd_per_1k"]), 8),
        }
    if observed_usd_per_1k is not None:
        updated["observed_cost_samples"] = observed_samples
        updated["mean_observed_usd_per_1k"] = round(cast(float, mean_observed), 8)
        updated["last_observed_usd_per_1k"] = round(observed_usd_per_1k, 8)
    elif observed_samples > 0:
        updated["observed_cost_samples"] = observed_samples
        updated["mean_observed_usd_per_1k"] = round(prev_mean_observed, 8)

    promo = evaluate_promotion(updated, cfg)
    updated.update(promo)

    await _save_model_stats(rds, model, updated)
    await _increment_daily(rds, cost_usd if success else 0.0)

    promotion_result: Dict[str, Any] = {"auto_promoted": False, "model": model}
    if promo["promotable"] and success:
        if rds:
            try:
                raw = await rds.get(REDIS_SUGGESTIONS_KEY)
                suggestions = json.loads(raw) if raw else []
                if not isinstance(suggestions, list):
                    suggestions = []
                if not any(s.get("model") == model for s in suggestions if isinstance(s, dict)):
                    suggestions.append(
                        {
                            "model": model,
                            "mean_reward": updated["mean_reward"],
                            "mean_latency_s": updated.get("mean_latency_s"),
                            "mean_observed_usd_per_1k": updated.get("mean_observed_usd_per_1k"),
                            "catalog_usd_per_1k": updated.get("catalog_usd_per_1k"),
                            "count": count,
                            "suggested_at": time.time(),
                            "promotion_passed": promo.get("promotion_passed"),
                        }
                    )
                    await rds.set(REDIS_SUGGESTIONS_KEY, json.dumps(suggestions[-50:]))
            except Exception as exc:
                logger.warning("[openrouter_explore] suggestion update failed: %s", exc)

        if cfg.auto_promote_enabled and not updated.get("auto_promoted_at"):
            promoted = await auto_promote_to_candidates(model, settings)
            if promoted:
                updated["auto_promoted_at"] = time.time()
                promotion_result["auto_promoted"] = True
                await _save_model_stats(rds, model, updated)
                _persist_stats_to_db(model, updated, auto_promoted=True)
            else:
                _persist_stats_to_db(model, updated)
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


async def maybe_run_shadow_comparison(
    *,
    query: str,
    explored_model: str,
    explored_answer: str,
    explored_quality: float,
    explored_latency: float,
    explored_cost: float,
    incumbent_model: Optional[str],
    deps: Dict[str, Any],
    settings: Any,
) -> None:
    """Run incumbent model on the same query and store paired comparison deltas."""
    if not incumbent_model or incumbent_model == explored_model:
        return
    if not incumbent_model.startswith(("openrouter/", "openai/", "anthropic/", "gemini/", "ollama/")):
        return

    try:
        out, meta = await deps["call_model"](
            model=incumbent_model,
            prompt=query,
            modality="text",
            image_b64=None,
            temperature=float(getattr(settings, "TEMPERATURE_DEFAULT", 0.2) or 0.2),
            max_tokens=int(getattr(settings, "MAX_TOKENS_DEFAULT", 512) or 512),
        )
        incumbent_answer = out if isinstance(out, str) else str(out)
        p_tok, c_tok, incumbent_cost, _, _ = deps["parse_meta_cost"](
            meta=meta,
            chosen_model=incumbent_model,
            cost_lookup=deps["get_model_cost"],
        )
        judge_scores = await deps["judge_answer"](query, incumbent_answer)
        valid_scores = [s["score"] for s in judge_scores if "score" in s]
        import numpy as np

        incumbent_quality = round((float(np.mean(valid_scores)) if valid_scores else 5.0) * 10.0, 2)

        delta = {
            "delta_quality": round(explored_quality - incumbent_quality, 3),
            "delta_latency_s": round(explored_latency - 0.0, 3),
            "delta_cost_usd": round(explored_cost - float(incumbent_cost or 0.0), 6),
            "incumbent_model": incumbent_model,
            "explored_model": explored_model,
            "recorded_at": time.time(),
        }

        rds = await _get_redis()
        if rds:
            stats = await _load_model_stats(rds, explored_model)
            shadows = stats.get("shadow_comparisons") or []
            if not isinstance(shadows, list):
                shadows = []
            shadows.append(delta)
            stats["shadow_comparisons"] = shadows[-20:]
            stats["shadow_delta_quality_mean"] = round(
                sum(float(s.get("delta_quality", 0)) for s in stats["shadow_comparisons"]) / len(stats["shadow_comparisons"]),
                4,
            )
            await _save_model_stats(rds, explored_model, stats)

        logger.info(
            "[openrouter_explore] shadow %s vs %s delta_q=%.2f delta_cost=%.6f",
            explored_model,
            incumbent_model,
            delta["delta_quality"],
            delta["delta_cost_usd"],
        )
    except Exception as exc:
        logger.warning("[openrouter_explore] shadow comparison failed: %s", exc)


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
