# -*- coding: utf-8 -*-
# Objective: Pure policy for OpenRouter exploration (config, pricing, promotion gate, UCB pick).
"""Exploration policy without I/O.

Configuration parsing, per-1k pricing helpers, the multi-objective promotion
gate (reward, latency, cost, reliability), the uncertainty-adaptive
exploration rate, budget/provider filters and the UCB pick over the pool.
Extracted from ``app.openrouter_explorer`` (re-exported there).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

from app.openrouter_catalog import get_openrouter_pricing_per_1k


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


def next_exploration_stats(
    stats: Dict[str, Any],
    *,
    reward: float,
    latency_s: float,
    cost_usd: float,
    prompt_tokens: int,
    completion_tokens: int,
    success: bool,
    judge_quality: Optional[float],
    catalog_usd_per_1k: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    """Running means, failure counters and cost observations after one exploration outcome."""
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

    return updated
