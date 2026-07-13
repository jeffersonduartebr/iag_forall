# Objective: Test async Redis hot-path integrations (phase 3).
"""Regression tests for async Redis bandit/settings/resilience paths."""

from __future__ import annotations

import pytest
from app.services import router_resilience as rr

from app import bandits


@pytest.mark.asyncio
async def test_get_ctx_stats_async_uses_redis_map(monkeypatch):
    async def _hgetall(_key):
        return {"m1": '{"mean": 0.8, "count": 3, "var": 0.1, "M2": 0.2, "alpha": 3.4, "beta": 1.0}'}

    monkeypatch.setattr(bandits, "redis_hgetall_map", _hgetall)
    stats = await bandits._get_ctx_stats_async("global")
    assert "m1" in stats
    assert stats["m1"]["mean"] == 0.8


@pytest.mark.asyncio
async def test_select_model_async_returns_choice(monkeypatch):
    async def _labels(query, modality="text"):
        return ["global"]

    async def _stats(ctx):
        return {"m1": {"mean": 0.5, "count": 1, "var": 0.0, "M2": 0.0, "alpha": 1.5, "beta": 1.0}}

    async def _combine(models, ctx_stats):
        return "m1", {"epsilon_greedy": "m1", "ucb1": "m1", "thompson": "m1"}

    monkeypatch.setattr(bandits, "_auto_context_labels_async", _labels)
    monkeypatch.setattr(bandits, "_get_ctx_stats_async", _stats)
    monkeypatch.setattr(bandits, "_meta_combine_choices_async", _combine)

    chosen = await bandits.select_model_async(["m1", "m2"], "hello", modality="text")
    assert chosen == "m1"


@pytest.mark.asyncio
async def test_settings_get_async_uses_async_redis(monkeypatch):
    from app import settings_dynamic as sd

    async def _redis_get(key):
        return "42" if key == "NSGA_W_QUALITY" else None

    monkeypatch.setattr(sd, "_get_from_redis_async", _redis_get)
    monkeypatch.setattr(sd, "_get_from_db", lambda key: None)
    sd._lru.clear()
    value = await sd.settings.get_async("NSGA_W_QUALITY", "1.0")
    assert value == "42"


@pytest.mark.asyncio
async def test_error_budget_async_reads_redis_hash(monkeypatch):
    async def _hgetall(_key):
        return {"total": "10", "errors": "3"}

    monkeypatch.setattr(rr, "redis_hgetall_map", _hgetall)

    def _settings_get(key, default=None):
        return {
            "ERROR_BUDGET_ENABLED": "1",
            "ERROR_BUDGET_WINDOW_S": "30",
            "ERROR_BUDGET_THRESHOLD": "0.2",
            "ERROR_BUDGET_MIN_REQUESTS": "5",
        }.get(key, default)

    exceeded = await rr.is_error_budget_exceeded_async(settings_getter=_settings_get)
    assert exceeded is True
