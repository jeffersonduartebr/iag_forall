# Objective: One Ollama warmup per deployment, whatever the number of uvicorn workers.
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_only_one_worker_warms_ollama_and_the_lock_is_released(monkeypatch, fake_redis):
    """Both uvicorn workers ran the warmup at once and loaded the GPU twice (Caso 1 cold-start storm)."""
    from app.services import ollama_preload as op

    runs = []

    async def _warm(models, configured):
        runs.append(fake_redis.get(op.WARMUP_LOCK_KEY) is not None)

    monkeypatch.setattr(op, "get_configured_ollama_warm_models", lambda: ["ollama/m"])
    monkeypatch.setattr(op, "pull_missing", AsyncMock())
    monkeypatch.setattr(op, "warm_runtimes", _warm)

    fake_redis.set(op.WARMUP_LOCK_KEY, "outro-worker")
    await op.preload_ollama_models()
    assert runs == []  # outro worker segura a trava: este não aquece

    fake_redis.delete(op.WARMUP_LOCK_KEY)
    await op.preload_ollama_models()
    assert runs == [True] and fake_redis.get(op.WARMUP_LOCK_KEY) is None  # aqueceu sob a trava e a soltou
    assert op.ollama_warmed_at() is not None


def _settings(monkeypatch, op, **listas):
    from types import SimpleNamespace

    monkeypatch.setattr(op, "settings", SimpleNamespace(**listas))
    monkeypatch.setattr(op, "VLM_OLLAMA_MODELS", [])
    monkeypatch.setattr(op, "get_configured_ollama_warm_models", lambda: [])


def test_ollama_is_in_use_only_when_a_route_or_judge_needs_it(monkeypatch):
    """Caso 1 without a GPU VM (Gemini + OpenRouter only): nothing may still depend on an absent daemon."""
    from app.services import ollama_preload as op

    _settings(monkeypatch, op, CANDIDATE_MODELS_LIST=["gemini/g", "openrouter/x/y"], JUDGE_MODELS=["gemini/g"])
    assert not op.ollama_in_use()
    _settings(monkeypatch, op, CANDIDATE_MODELS_LIST=["gemini/g"], JUDGE_MODELS=["ollama/granite"])
    assert op.ollama_in_use()


@pytest.mark.asyncio
async def test_no_local_model_means_no_pull_and_no_ollama_health_probe(monkeypatch):
    from app.services import ollama_preload as op

    from app import health

    _settings(monkeypatch, op, CANDIDATE_MODELS_LIST=["gemini/g"])
    pull = AsyncMock()
    monkeypatch.setattr(op, "pull_missing", pull)
    await op.preload_ollama_models()
    assert not pull.called

    async def _ok(name):
        return health.ComponentHealth(name=name, healthy=True)

    for probe in ("redis", "database", "vectorstore", "circuit_breakers"):
        monkeypatch.setattr(health, f"check_{probe}_health", lambda p=probe: _ok(p))
    monkeypatch.setattr(health, "check_ollama_health", AsyncMock(side_effect=AssertionError("sondou o Ollama")))
    result = await health.get_full_health_check(force_refresh=True)
    assert result["status"] == "healthy" and "ollama" not in result["components"]
