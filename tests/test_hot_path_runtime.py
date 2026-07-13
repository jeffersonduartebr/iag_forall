# Objective: Test hot-path async/offload helpers for governance and routing.
"""Regression tests for second-round hot-path async adapters."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from app.services import hot_path_cache as hpc
from app.services import hot_path_runtime as hpr


@pytest.mark.asyncio
async def test_get_active_policy_async_uses_thread_offload(monkeypatch):
    calls = {"count": 0}

    def _policy():
        calls["count"] += 1
        return {"version": "v1", "config": {}}

    async def _redis_get(_key):
        return None

    async def _redis_set(*_args, **_kwargs):
        return None

    monkeypatch.setattr(hpr, "get_active_policy", _policy)
    monkeypatch.setattr(hpr, "_redis_get_json", _redis_get)
    monkeypatch.setattr(hpr, "_redis_set_json", _redis_set)

    out = await hpr.get_active_policy_async()
    assert out["version"] == "v1"
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_check_tenant_budget_async_offloads_sync_helper(monkeypatch):
    expected = SimpleNamespace(allowed=True, reason="ok")

    monkeypatch.setattr(hpr, "check_tenant_budget", lambda tenant_id, projected_cost_usd=0.0: expected)
    out = await hpr.check_tenant_budget_async("tenant-a", 0.01)
    assert out is expected


@pytest.mark.asyncio
async def test_choose_top2_models_async_offloads_sync_helper(monkeypatch):
    monkeypatch.setattr(hpr, "choose_top2_models", lambda *args, **kwargs: ["openai/gpt-4o", "ollama/phi4:latest"])
    out = await hpr.choose_top2_models_async(
        candidates=["openai/gpt-4o", "ollama/phi4:latest"],
        weights={"w_quality": 1.0, "w_latency": 0.5, "w_cost": 50.0},
        query_text="hello",
        modality="text",
        uncertainty_score=0.2,
    )
    assert out == ["openai/gpt-4o", "ollama/phi4:latest"]


def test_schedule_tenant_usage_falls_back_without_running_loop(monkeypatch):
    calls = []

    monkeypatch.setattr(hpr, "record_tenant_usage", lambda tenant_id, **kwargs: calls.append((tenant_id, kwargs)))
    hpr.schedule_tenant_usage(tenant_id="t1", cost_usd=0.5, tokens_in=1, tokens_out=2, requests=1)
    assert calls == [("t1", {"cost_usd": 0.5, "tokens_in": 1, "tokens_out": 2, "requests": 1})]


@pytest.mark.asyncio
async def test_schedule_tenant_usage_queues_background_task(monkeypatch):
    calls = []

    def _record(tenant_id, **kwargs):
        calls.append((tenant_id, kwargs))

    monkeypatch.setattr(hpr, "record_tenant_usage", _record)
    hpr.schedule_tenant_usage(tenant_id="t2", cost_usd=0.2, tokens_in=3, tokens_out=4, requests=1)
    await asyncio.sleep(0.05)
    assert calls == [("t2", {"cost_usd": 0.2, "tokens_in": 3, "tokens_out": 4, "requests": 1})]


def test_ttl_cache_expires_entries(monkeypatch):
    cache = hpc.TTLCache(ttl_s=0.01)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    import time

    time.sleep(0.02)
    assert not hpc.cache_hit(cache.get("k"))
