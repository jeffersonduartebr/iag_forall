# Objective: Test OpenRouter catalog exploration module.
"""Tests for OpenRouter exploration picker and stats."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from app import openrouter_explorer as ore


class _Settings:
    def __init__(self, **kwargs):
        self._data = kwargs
        self._lists = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value, actor="system", source="internal"):
        if key == "CANDIDATE_MODELS_LIST":
            self._lists["CANDIDATE_MODELS_LIST"] = list(value)
            self._data["CANDIDATE_MODELS_LIST"] = value
        else:
            self._data[key] = value

    @property
    def CANDIDATE_MODELS_LIST(self):
        return self._lists.get("CANDIDATE_MODELS_LIST", self._data.get("CANDIDATE_MODELS_LIST", []))


@pytest.fixture
def settings_enabled():
    return _Settings(
        OPENROUTER_EXPLORATION_ENABLED="1",
        OPENROUTER_EXPLORATION_MODE="balanced",
        OPENROUTER_EXPLORATION_RATE="1.0",
        OPENROUTER_EXPLORATION_MAX_PER_DAY="50",
        OPENROUTER_EXPLORATION_MAX_USD_PER_DAY="10.0",
        OPENROUTER_EXPLORATION_MAX_PRICE_PROMPT_1K="0.05",
        OPENROUTER_EXPLORATION_MAX_PRICE_COMPLETION_1K="0.05",
        OPENROUTER_EXPLORATION_PROMOTE_MIN_SAMPLES="5",
        OPENROUTER_EXPLORATION_PROMOTE_MIN_REWARD="0.7",
        OPENROUTER_EXPLORATION_PROMOTE_MAX_LATENCY_S="30.0",
        OPENROUTER_EXPLORATION_PROMOTE_MAX_COST_USD_PER_1K="0.02",
        OPENROUTER_EXPLORATION_AUTO_PROMOTE_ENABLED="1",
        OPENROUTER_EXPLORATION_ADAPTIVE_RATE_ENABLED="1",
        OPENROUTER_EXPLORATION_SHADOW_COMPARE_RATE="0.0",
        OPENROUTER_EXPLORATION_PROVIDER_ALLOWLIST='["anthropic","openai"]',
        OPENROUTER_EXPLORATION_POOL_SIZE="10",
        CANDIDATE_MODELS_LIST=["openrouter/openai/gpt-4o-mini"],
    )


def test_load_exploration_config(settings_enabled):
    cfg = ore.load_exploration_config(settings_enabled)
    assert cfg.enabled is True
    assert cfg.rate == 1.0
    assert cfg.auto_promote_enabled is True
    assert cfg.mode == "balanced"
    assert "anthropic" in cfg.provider_allowlist


def test_evaluate_promotion_multi_objective(settings_enabled):
    cfg = ore.load_exploration_config(settings_enabled)
    good = {
        "count": 20,
        "mean_reward": 0.8,
        "mean_latency_s": 2.0,
        "mean_observed_usd_per_1k": 0.001,
        "failure_count": 0,
        "mean_judge_quality": 7.2,
    }
    out = ore.evaluate_promotion(good, cfg)
    assert out["promotable"] is True
    assert not out["promotion_blockers"]
    assert "eval_gate_ok" in out["promotion_passed"]

    bad = {**good, "mean_latency_s": 45.0}
    out_bad = ore.evaluate_promotion(bad, cfg)
    assert out_bad["promotable"] is False
    assert "latencia_alta" in out_bad["promotion_blockers"]

    no_judge = {**good, "mean_judge_quality": None}
    del no_judge["mean_judge_quality"]
    out_judge = ore.evaluate_promotion(no_judge, cfg)
    assert out_judge["promotable"] is False
    assert "eval_gate_sem_judge" in out_judge["promotion_blockers"]


def test_effective_exploration_rate_adaptive(settings_enabled):
    cfg = ore.load_exploration_config(settings_enabled)
    low = ore._effective_exploration_rate(cfg, 0.0)
    high = ore._effective_exploration_rate(cfg, 1.0)
    assert high > low


@pytest.mark.asyncio
async def test_maybe_pick_returns_openrouter_model(settings_enabled, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    catalog = [
        {
            "id": "openai/gpt-4o-mini",
            "full_name": "openrouter/openai/gpt-4o-mini",
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
        },
        {
            "id": "anthropic/claude-3.5-haiku",
            "full_name": "openrouter/anthropic/claude-3.5-haiku",
            "pricing": {"prompt": "0.0000008", "completion": "0.000004"},
        },
    ]

    class _Rds:
        async def get(self, key):
            return None

        async def set(self, key, value):
            return True

        async def incr(self, key):
            return 1

        async def expire(self, key, ttl):
            return True

        async def sadd(self, key, value):
            return 1

    monkeypatch.setattr(ore, "fetch_openrouter_models", AsyncMock(return_value=catalog))
    monkeypatch.setattr(ore, "_get_redis", AsyncMock(return_value=_Rds()))
    monkeypatch.setattr(ore, "_get_blocklist", AsyncMock(return_value=set()))
    monkeypatch.setattr(ore.random, "random", lambda: 0.0)

    result = await ore.maybe_pick_exploration_model(
        known_models={"openrouter/openai/gpt-4o-mini"},
        modality="text",
        settings=settings_enabled,
        uncertainty_score=0.8,
    )
    assert result is not None
    model, meta = result
    assert model.startswith("openrouter/")
    assert model != "openrouter/openai/gpt-4o-mini"
    assert meta.get("openrouter_exploration") is True
    assert meta.get("adaptive_rate") is True


@pytest.mark.asyncio
async def test_maybe_pick_skips_when_disabled(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    settings = _Settings(OPENROUTER_EXPLORATION_ENABLED="0")
    result = await ore.maybe_pick_exploration_model(
        known_models=set(),
        modality="text",
        settings=settings,
    )
    assert result is None


@pytest.mark.asyncio
async def test_record_exploration_outcome_updates_stats(settings_enabled, monkeypatch):
    store = {}

    class _Rds:
        async def get(self, key):
            return store.get(key)

        async def set(self, key, value):
            store[key] = value

        async def sadd(self, key, value):
            return 1

        async def incr(self, key):
            return 1

        async def incrbyfloat(self, key, value):
            return float(value)

        async def expire(self, key, ttl):
            return True

    monkeypatch.setattr(ore, "_get_redis", AsyncMock(return_value=_Rds()))
    monkeypatch.setattr(ore, "_persist_stats_to_db", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ore,
        "_catalog_usd_per_1k_for_model",
        lambda _model: {"prompt_usd_per_1k": 0.00015, "completion_usd_per_1k": 0.0006},
    )

    await ore.record_exploration_outcome(
        model="openrouter/anthropic/claude-3.5-haiku",
        reward=0.8,
        latency_s=1.2,
        cost_usd=0.001,
        settings=settings_enabled,
        prompt_tokens=1000,
        completion_tokens=500,
        success=True,
    )

    key = f"{ore.REDIS_MODEL_STATS_PREFIX}openrouter/anthropic/claude-3.5-haiku"
    assert key in store
    stats = json.loads(store[key])
    assert stats["count"] == 1
    assert stats["mean_reward"] == pytest.approx(0.8)
    assert stats["mean_latency_s"] == pytest.approx(1.2)
    assert stats["mean_observed_usd_per_1k"] == pytest.approx(0.00066667, rel=0, abs=1e-8)


@pytest.mark.asyncio
async def test_auto_promote_adds_candidate(settings_enabled, monkeypatch):
    # O eval gate (WIP) exige qualidade de juiz; este teste cobre a promoção
    # multi-objetivo pura, então desabilita o gate localmente.
    settings_enabled.set("OPENROUTER_EXPLORATION_PROMOTE_EVAL_GATE_ENABLED", "0")
    store = {}

    class _Rds:
        async def get(self, key):
            return store.get(key)

        async def set(self, key, value):
            store[key] = value

        async def sadd(self, key, value):
            return 1

        async def incr(self, key):
            return 1

        async def incrbyfloat(self, key, value):
            return float(value)

        async def expire(self, key, ttl):
            return True

    monkeypatch.setattr(ore, "_get_redis", AsyncMock(return_value=_Rds()))
    monkeypatch.setattr(ore, "_persist_stats_to_db", lambda *args, **kwargs: None)
    monkeypatch.setattr(ore, "invalidate_exploration_pool_cache", AsyncMock())
    monkeypatch.setattr(
        ore,
        "_catalog_usd_per_1k_for_model",
        lambda _model: {"prompt_usd_per_1k": 0.00015, "completion_usd_per_1k": 0.0006},
    )

    model = "openrouter/anthropic/claude-3.5-haiku"
    for i in range(5):
        result = await ore.record_exploration_outcome(
            model=model,
            reward=0.85,
            latency_s=1.0,
            cost_usd=0.0005,
            settings=settings_enabled,
            prompt_tokens=500,
            completion_tokens=200,
            success=True,
        )

    assert result is not None
    assert result.get("auto_promoted") is True
    assert model in settings_enabled.CANDIDATE_MODELS_LIST


@pytest.mark.asyncio
async def test_get_exploration_status_adds_cost_comparison(settings_enabled, monkeypatch):
    store = {
        f"{ore.REDIS_MODEL_STATS_PREFIX}openrouter/openai/gpt-4o-mini": json.dumps(
            {
                "count": 3,
                "mean_reward": 0.75,
                "mean_observed_usd_per_1k": 0.0004,
                "observed_cost_samples": 3,
                "failure_count": 0,
                "catalog_usd_per_1k": {"prompt_usd_per_1k": 0.00015, "completion_usd_per_1k": 0.0006},
            }
        ),
        f"{ore.REDIS_MODEL_STATS_PREFIX}openrouter/anthropic/claude-3.5-haiku": json.dumps(
            {
                "count": 2,
                "mean_reward": 0.8,
                "mean_observed_usd_per_1k": 0.0012,
                "observed_cost_samples": 2,
                "failure_count": 0,
                "catalog_usd_per_1k": {"prompt_usd_per_1k": 0.0008, "completion_usd_per_1k": 0.004},
            }
        ),
    }

    class _Rds:
        async def get(self, key):
            return store.get(key)

        async def smembers(self, key):
            return [
                "openrouter/openai/gpt-4o-mini",
                "openrouter/anthropic/claude-3.5-haiku",
            ]

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(ore, "_get_redis", AsyncMock(return_value=_Rds()))
    monkeypatch.setattr(ore, "_get_daily_count", AsyncMock(return_value=0))
    monkeypatch.setattr(ore, "_get_daily_usd", AsyncMock(return_value=0.0))
    monkeypatch.setattr(ore, "_get_blocklist", AsyncMock(return_value=set()))

    status = await ore.get_exploration_status(settings_enabled)
    assert status["cost_benchmark"]["pool_median_observed_usd_per_1k"] == pytest.approx(0.0008)
    tiers = {m["model"]: m["cost_tier"] for m in status["models"]}
    assert tiers["openrouter/openai/gpt-4o-mini"] == "mais_barato"
    assert tiers["openrouter/anthropic/claude-3.5-haiku"] == "mais_caro"
    assert status["config"]["auto_promote_enabled"] is True


def test_apply_mode_preset():
    from app.openrouter_exploration_modes import apply_mode_preset

    preset = apply_mode_preset("cost_hunt")
    assert preset["OPENROUTER_EXPLORATION_RATE"] == "0.15"
