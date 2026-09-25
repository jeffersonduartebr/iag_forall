# Objective: OpenRouter exploration pick — gates, daily caps, pool eligibility, pool cache and blocklist.
"""``maybe_pick_exploration_model`` and the exploration pool against fakeredis (no network)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from app import openrouter_catalog as cat
from app import openrouter_explorer as ore

CATALOG = [
    {"id": "openai/cheap", "full_name": "openrouter/openai/cheap", "pricing": {"prompt": "0.0000001", "completion": "0.0000002"}},
    {"id": "openai/mid", "full_name": "openrouter/openai/mid", "pricing": {"prompt": "0.000002", "completion": "0.000004"}},
    {"id": "openai/pricey", "full_name": "openrouter/openai/pricey", "pricing": {"prompt": "0.001", "completion": "0.001"}},
    {"id": "openrouter/auto", "full_name": "openrouter/openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
    {"id": "shady/model", "full_name": "openrouter/shady/model", "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "openai/known", "full_name": "openrouter/openai/known", "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "local", "full_name": "ollama/local"},
]


KNOWN = frozenset({"openrouter/openai/known"})


def _settings(**overrides):
    base = {
        "OPENROUTER_EXPLORATION_ENABLED": "1", "OPENROUTER_EXPLORATION_RATE": "1.0",
        "OPENROUTER_EXPLORATION_ADAPTIVE_RATE_ENABLED": "0", "OPENROUTER_EXPLORATION_SHADOW_COMPARE_RATE": "0",
        "OPENROUTER_EXPLORATION_PROVIDER_ALLOWLIST": "openai,openrouter", "OPENROUTER_EXPLORATION_MAX_PER_DAY": "2",
        "OPENROUTER_EXPLORATION_MAX_USD_PER_DAY": "1.0", "OPENROUTER_EXPLORATION_POOL_CACHE_TTL_S": "600",
    }
    base.update(overrides)
    return base


@pytest.fixture
def env(monkeypatch, fake_aioredis, fake_clock):
    async def _rds():
        return fake_aioredis

    monkeypatch.setattr(ore, "_get_redis", _rds)
    monkeypatch.setattr(ore, "_openrouter_configured", lambda: True)
    monkeypatch.setattr(ore.random, "random", lambda: 0.0)
    monkeypatch.setitem(cat._CACHE, "data", CATALOG)
    fetch = AsyncMock(return_value=CATALOG)
    monkeypatch.setattr(ore, "fetch_openrouter_models", fetch)
    return type("Env", (), {"rds": fake_aioredis, "fetch": fetch, "clock": fake_clock(ore)})()


async def _pick(settings=None, known=KNOWN, modality="text", uq=0.5):
    return await ore.maybe_pick_exploration_model(
        known_models=set(known), modality=modality, settings=settings or _settings(), uncertainty_score=uq
    )


@pytest.mark.asyncio
async def test_pool_keeps_only_allowed_priced_unknown_openrouter_models(env):
    cfg = ore.load_exploration_config(_settings())
    pool = await ore._build_exploration_pool({"openrouter/openai/known"}, cfg, "text")
    # fora: caro, preço variável (-1), provedor fora da allowlist, já conhecido, não-OpenRouter
    assert sorted(pool) == ["openrouter/openai/cheap", "openrouter/openai/mid"]
    assert await ore._build_exploration_pool(KNOWN, cfg, "image") == []


@pytest.mark.asyncio
async def test_pick_returns_unseen_model_with_debug_metadata(env):
    model, meta = await _pick()
    assert model in {"openrouter/openai/cheap", "openrouter/openai/mid"}
    assert meta["openrouter_exploration"] is True and meta["exploration_pool_size"] == 2
    assert meta["exploration_rate"] == 1.0 and meta["shadow_compare"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "configured", "modality", "draw"),
    [(_settings(OPENROUTER_EXPLORATION_ENABLED="0"), True, "text", 0.0), (_settings(), False, "text", 0.0),
     (_settings(), True, "image", 0.0), (_settings(OPENROUTER_EXPLORATION_RATE="0.2"), True, "text", 0.5)],
)
async def test_pick_declines_when_off_unconfigured_non_text_or_not_sampled(env, monkeypatch, settings, configured,
                                                                           modality, draw):
    monkeypatch.setattr(ore, "_openrouter_configured", lambda: configured)
    monkeypatch.setattr(ore.random, "random", lambda: draw)
    assert await _pick(settings, modality=modality) is None
    env.fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_daily_count_cap_stops_exploration(env):
    for _ in range(2):
        await ore._increment_daily(env.rds, 0.01)
    assert await _pick() is None
    env.fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_daily_usd_cap_stops_exploration_but_zero_means_unlimited(env):
    await ore._increment_daily(env.rds, 1.5)
    assert await _pick() is None
    assert await _pick(_settings(OPENROUTER_EXPLORATION_MAX_USD_PER_DAY="0")) is not None


@pytest.mark.asyncio
async def test_pool_is_cached_until_ttl_then_rebuilt(env):
    cfg = ore.load_exploration_config(_settings())
    first = await ore._build_exploration_pool(KNOWN, cfg, "text")
    env.fetch.return_value = [CATALOG[1]]
    assert await ore._build_exploration_pool(KNOWN, cfg, "text") == first
    # o cache ainda respeita modelos que viraram conhecidos
    assert "openrouter/openai/cheap" not in await ore._build_exploration_pool(KNOWN | {"openrouter/openai/cheap"}, cfg, "text")
    assert env.fetch.await_count == 1
    env.clock.advance(601)
    assert await ore._build_exploration_pool(KNOWN, cfg, "text") == ["openrouter/openai/mid"]
    assert env.fetch.await_count == 2


@pytest.mark.asyncio
async def test_blocklisting_invalidates_the_cached_pool(env):
    cfg = ore.load_exploration_config(_settings())
    await ore._build_exploration_pool(KNOWN, cfg, "text")
    await ore.blocklist_model("openrouter/openai/cheap")
    assert await env.rds.get(ore.REDIS_POOL_CACHE_KEY) is None
    assert await ore._build_exploration_pool(KNOWN, cfg, "text") == ["openrouter/openai/mid"]
    await ore.unblocklist_model("openrouter/openai/cheap")
    assert await ore._get_blocklist(env.rds) == set()
    assert "openrouter/openai/cheap" in await ore._build_exploration_pool(KNOWN, cfg, "text")


@pytest.mark.asyncio
async def test_models_failing_repeatedly_or_already_promoted_leave_the_pool(env):
    await ore._save_model_stats(env.rds, "openrouter/openai/cheap", {"count": 3, "consecutive_failures": 3})
    promoted = {"count": 20, "mean_reward": 0.95, "mean_latency_s": 1.0, "mean_judge_quality": 9.0,
                "auto_promoted_at": 1.0}
    await ore._save_model_stats(env.rds, "openrouter/openai/mid", promoted)
    cfg = ore.load_exploration_config(_settings())
    assert await ore._build_exploration_pool(KNOWN, cfg, "text") == []
    assert await _pick() is None


@pytest.mark.asyncio
async def test_cost_hunt_prefers_cheaper_models_at_equal_observations(env):
    cfg = ore.load_exploration_config(_settings(OPENROUTER_EXPLORATION_MODE="cost_hunt"))
    assert await ore._build_exploration_pool(KNOWN, cfg, "text") == ["openrouter/openai/cheap", "openrouter/openai/mid"]
    balanced = ore.load_exploration_config(_settings())
    assert ore._explore_score("openrouter/openai/mid", {}, balanced) == 1.0


@pytest.mark.asyncio
async def test_empty_catalog_or_corrupt_cache_yield_a_rebuilt_or_empty_pool(env):
    await env.rds.set(ore.REDIS_POOL_CACHE_KEY, "{corrupt")
    env.fetch.return_value = []
    assert await ore._build_exploration_pool(KNOWN, ore.load_exploration_config(_settings()), "text") == []
    assert await ore._cached_pool(None, set(), set()) is None
    await env.rds.set(ore.REDIS_POOL_CACHE_KEY, json.dumps({"pool": "x", "expires_at": 1e12}))
    assert await ore._cached_pool(env.rds, set(), set()) is None


@pytest.mark.asyncio
async def test_blocklist_helpers_are_noops_without_redis(monkeypatch):
    monkeypatch.setattr(ore, "_get_redis", AsyncMock(return_value=None))
    await ore.blocklist_model("m")
    await ore.unblocklist_model("m")
    await ore.invalidate_exploration_pool_cache()


@pytest.mark.asyncio
async def test_redis_and_configuration_probes(monkeypatch):
    sentinel = object()
    monkeypatch.setattr("app.utils.redis_client.get_redis_async_safe", lambda: sentinel)
    assert await ore._get_redis() is sentinel
    monkeypatch.setattr(cat, "get_openrouter_api_key", lambda: "sk-x")
    assert ore._openrouter_configured() is True
