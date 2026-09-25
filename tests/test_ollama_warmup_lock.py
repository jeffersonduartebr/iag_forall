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
