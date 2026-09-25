# Objective: OpenRouter exploration outcomes — stats, daily budget, suggestions, auto-promotion, status, DB persist.
"""``record_exploration_outcome`` / ``auto_promote_to_candidates`` / ``get_exploration_status`` on fakeredis."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from app import openrouter_catalog as cat
from app import openrouter_explorer as ore

MODEL = "openrouter/acme/fast"


class _Settings(dict):
    """Dynamic-settings stand-in: ``get`` from the dict, ``set`` records the write."""

    def set(self, key, value, actor="system", source="internal"):
        self[key] = value

    @property
    def CANDIDATE_MODELS_LIST(self):
        return self.get("CANDIDATE_MODELS_LIST", [])


def _settings(**overrides):
    base = {"OPENROUTER_EXPLORATION_ENABLED": "1", "OPENROUTER_EXPLORATION_PROMOTE_MIN_SAMPLES": "2",
            "OPENROUTER_EXPLORATION_PROMOTE_MIN_REWARD": "0.7", "OPENROUTER_EXPLORATION_MAX_USD_PER_DAY": "2.0",
            "CANDIDATE_MODELS_LIST": ["ollama/llama3"]}
    base.update(overrides)
    return _Settings(base)


@pytest.fixture
def env(monkeypatch, fake_aioredis):
    async def _rds():
        return fake_aioredis

    persisted = []
    monkeypatch.setattr(ore, "_get_redis", _rds)
    monkeypatch.setattr(ore, "_openrouter_configured", lambda: True)
    monkeypatch.setattr(ore, "_persist_stats_to_db", lambda m, s, auto_promoted=False: persisted.append((m, auto_promoted)))
    monkeypatch.setitem(cat._CACHE, "data", [{"id": "acme/fast", "pricing": {"prompt": "0.000001", "completion": "0.000001"}}])
    return type("Env", (), {"rds": fake_aioredis, "persisted": persisted})()


async def _outcome(settings, *, model=MODEL, success=True, judge=8.0, reward=0.9, cost=0.0005):
    return await ore.record_exploration_outcome(model=model, reward=reward, latency_s=1.5, cost_usd=cost,
                                                settings=settings, prompt_tokens=400, completion_tokens=100,
                                                success=success, judge_quality=judge)


@pytest.mark.asyncio
async def test_non_openrouter_models_are_ignored(env):
    assert await _outcome(_settings(), model="ollama/llama3") is None
    assert env.persisted == []


@pytest.mark.asyncio
async def test_good_model_is_suggested_then_auto_promoted_once(env):
    settings = _settings()
    first = await _outcome(settings)
    assert first["promotable"] is False and first["promotion_blockers"] == ["amostras_insuficientes"]
    second = await _outcome(settings)
    assert second == {"auto_promoted": True, "model": MODEL, "promotable": True, "promotion_blockers": []}
    assert settings["CANDIDATE_MODELS_LIST"] == ["ollama/llama3", MODEL]
    assert [s["model"] for s in json.loads(await env.rds.get(ore.REDIS_SUGGESTIONS_KEY))] == [MODEL]
    assert [p["model"] for p in json.loads(await env.rds.get(ore.REDIS_AUTO_PROMOTED_KEY))] == [MODEL]
    stats = await ore._load_model_stats(env.rds, MODEL)
    assert stats["auto_promoted_at"] and stats["count"] == 2 and stats["catalog_usd_per_1k"]
    assert env.persisted[-1] == (MODEL, True)
    assert await ore._get_daily_count(env.rds) == 2
    assert await ore._get_daily_usd(env.rds) == pytest.approx(0.001)


@pytest.mark.asyncio
async def test_admin_removal_of_a_promoted_model_is_respected(env):
    """Regression: the promotion mark was lost on the next outcome, so the model was re-added to the candidates."""
    settings = _settings()
    await _outcome(settings)
    await _outcome(settings)
    settings["CANDIDATE_MODELS_LIST"] = ["ollama/llama3"]  # admin tirou o modelo da lista
    third = await _outcome(settings)
    assert third["auto_promoted"] is False and third["promotable"] is True
    assert settings["CANDIDATE_MODELS_LIST"] == ["ollama/llama3"]
    assert (await ore._load_model_stats(env.rds, MODEL))["auto_promoted_at"]
    assert len(json.loads(await env.rds.get(ore.REDIS_SUGGESTIONS_KEY))) == 1  # sem sugestão duplicada


@pytest.mark.asyncio
async def test_auto_promote_disabled_only_suggests(env):
    settings = _settings(OPENROUTER_EXPLORATION_AUTO_PROMOTE_ENABLED="0")
    await _outcome(settings)
    result = await _outcome(settings)
    assert result["auto_promoted"] is False and result["promotable"] is True
    assert settings["CANDIDATE_MODELS_LIST"] == ["ollama/llama3"]
    assert await env.rds.get(ore.REDIS_SUGGESTIONS_KEY) is not None
    assert env.persisted[-1] == (MODEL, False)


@pytest.mark.asyncio
async def test_failures_count_toward_blocking_but_not_toward_spend(env, caplog):
    caplog.set_level("INFO", logger="app.openrouter_explorer")
    settings = _settings(OPENROUTER_EXPLORATION_CONSECUTIVE_FAILURE_BLOCK="2")
    for _ in range(2):
        await ore.record_exploration_failure(model=MODEL, settings=settings, error="HTTP 502")
    stats = await ore._load_model_stats(env.rds, MODEL)
    assert stats["consecutive_failures"] == 2 and stats["failure_rate"] == 1.0
    assert await ore._get_daily_usd(env.rds) == 0.0 and await ore._get_daily_count(env.rds) == 2
    assert ore._explore_score(MODEL, stats, ore.load_exploration_config(settings)) is None
    assert "HTTP 502" in caplog.text


@pytest.mark.asyncio
async def test_auto_promote_guards(env):
    assert await ore.auto_promote_to_candidates(MODEL, _settings(OPENROUTER_EXPLORATION_AUTO_PROMOTE_ENABLED="0")) is False
    assert await ore.auto_promote_to_candidates("ollama/x", _settings()) is False
    assert await ore.auto_promote_to_candidates(MODEL, _settings(CANDIDATE_MODELS_LIST=[MODEL])) is False


@pytest.mark.asyncio
async def test_auto_promote_with_callable_getter_and_corrupt_history(env):
    class _Callable(_Settings):
        def CANDIDATE_MODELS_LIST(self):  # noqa: N802  (espelha a API de settings_dynamic)
            return self.get("CANDIDATE_MODELS_LIST", [])

    settings = _Callable(_settings())
    await env.rds.set(ore.REDIS_AUTO_PROMOTED_KEY, json.dumps({"not": "a list"}))
    assert await ore.auto_promote_to_candidates(MODEL, settings) is True
    assert settings["CANDIDATE_MODELS_LIST"][-1] == MODEL
    assert [p["model"] for p in json.loads(await env.rds.get(ore.REDIS_AUTO_PROMOTED_KEY))] == [MODEL]


@pytest.mark.asyncio
async def test_suggestion_list_recovers_from_corrupt_payload(env):
    await env.rds.set(ore.REDIS_SUGGESTIONS_KEY, json.dumps({"x": 1}))
    await ore._record_suggestion(env.rds, MODEL, {"mean_reward": 0.9, "count": 3}, {"promotion_passed": ["reward_ok"]})
    assert [s["model"] for s in json.loads(await env.rds.get(ore.REDIS_SUGGESTIONS_KEY))] == [MODEL]
    await env.rds.set(ore.REDIS_SUGGESTIONS_KEY, "{bad")
    await ore._record_suggestion(env.rds, MODEL, {"mean_reward": 0.9, "count": 3}, {})
    await ore._record_suggestion(None, MODEL, {}, {})
    assert await env.rds.get(ore.REDIS_SUGGESTIONS_KEY) == b"{bad"


@pytest.mark.asyncio
async def test_status_reports_usage_models_and_blocklist(env):
    settings = _settings()
    await _outcome(settings)
    await ore._save_model_stats(env.rds, "openrouter/acme/slow", {"count": 1, "mean_reward": 0.2})
    await env.rds.sadd(ore.REDIS_MODELS_SET_KEY, "openrouter/ghost")
    await ore.blocklist_model("openrouter/acme/slow")
    status = await ore.get_exploration_status(settings)
    assert status["enabled"] is True and status["usage"]["explorations_today"] == 1
    assert status["usage"]["usd_remaining_today"] == pytest.approx(2.0 - 0.0005)
    assert [m["model"] for m in status["models"]] == [MODEL, "openrouter/acme/slow"]
    assert status["models"][1]["blocklisted"] is True and status["models"][1].get("catalog_usd_per_1k") is None
    assert status["blocklist"] == ["openrouter/acme/slow"] and status["suggestions"] == []


@pytest.mark.asyncio
async def test_status_fills_catalog_price_and_tolerates_corrupt_lists(env):
    await ore._save_model_stats(env.rds, MODEL, {"count": 1, "mean_reward": 0.5})
    await env.rds.set(ore.REDIS_SUGGESTIONS_KEY, "{bad")
    await env.rds.set(ore.REDIS_AUTO_PROMOTED_KEY, "{bad")
    status = await ore.get_exploration_status(_settings(OPENROUTER_EXPLORATION_MAX_USD_PER_DAY="0"))
    assert status["models"][0]["catalog_usd_per_1k"] == {"prompt_usd_per_1k": 0.001, "completion_usd_per_1k": 0.001}
    assert status["suggestions"] == [] and status["auto_promoted"] == []
    assert status["usage"]["usd_remaining_today"] is None


@pytest.mark.asyncio
async def test_status_without_redis(monkeypatch):
    monkeypatch.setattr(ore, "_get_redis", AsyncMock(return_value=None))
    monkeypatch.setattr(ore, "_openrouter_configured", lambda: False)
    status = await ore.get_exploration_status(_settings())
    assert status["enabled"] is False and status["setting_enabled"] is True
    assert status["usage"]["remaining_today"] == 100 and status["models"] == []


def test_persist_upserts_stats_and_swallows_db_errors(monkeypatch):
    calls = []

    class _Conn:
        def execute(self, stmt, params):
            calls.append(params)

    class _Engine:
        def begin(self):
            return self

        def __enter__(self):
            return _Conn()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("app.db.get_engine", lambda: _Engine())
    stats = {"count": 3, "mean_reward": 0.8, "catalog_usd_per_1k": {"prompt_usd_per_1k": 0.1}}
    ore._persist_stats_to_db(MODEL, stats, auto_promoted=True)
    assert calls[0]["model"] == MODEL and calls[0]["auto_promoted"] == 1
    assert calls[0]["catalog_prompt_usd_per_1k"] == 0.1 and json.loads(calls[0]["stats_json"]) == stats

    def _down():
        raise RuntimeError("db down")

    monkeypatch.setattr("app.db.get_engine", _down)
    ore._persist_stats_to_db(MODEL, stats)  # não propaga
