# Objective: Tests for the dedicated CPU-bound executor and async embed wrappers (perf #23).
"""Tests for utils.executors and the embeddings async wrappers."""

from __future__ import annotations

import concurrent.futures
import threading

import pytest
from app.utils import executors


@pytest.fixture(autouse=True)
def _reset_pool():
    """Ensure each test starts and ends without a lingering global pool."""
    executors.shutdown_cpu_executor()
    yield
    executors.shutdown_cpu_executor()


def test_resolve_workers_env_override(monkeypatch):
    """A positive EMBED_CPU_THREADS overrides the CPU-count default."""
    monkeypatch.setenv("EMBED_CPU_THREADS", "7")
    assert executors._resolve_workers() == 7


def test_resolve_workers_invalid_env_falls_back(monkeypatch):
    """A non-numeric override is ignored and the clamped default (2..4) is used."""
    monkeypatch.setenv("EMBED_CPU_THREADS", "not-a-number")
    workers = executors._resolve_workers()
    assert 2 <= workers <= 4


def test_get_cpu_executor_is_singleton():
    """The pool is created once and reused across calls."""
    first = executors.get_cpu_executor()
    second = executors.get_cpu_executor()
    assert first is second
    assert isinstance(first, concurrent.futures.ThreadPoolExecutor)


def test_shutdown_resets_singleton():
    """Shutdown clears the global so a fresh pool is built on next use."""
    first = executors.get_cpu_executor()
    executors.shutdown_cpu_executor()
    second = executors.get_cpu_executor()
    assert first is not second


@pytest.mark.asyncio
async def test_run_cpu_bound_executes_off_the_event_loop_thread():
    """run_cpu_bound returns the callable result from a worker thread, not the loop thread."""
    loop_thread = threading.get_ident()
    seen = {}

    def _work(a, b):
        seen["thread"] = threading.get_ident()
        return a + b

    result = await executors.run_cpu_bound(_work, 2, 3)
    assert result == 5
    assert seen["thread"] != loop_thread


@pytest.mark.asyncio
async def test_aembed_text_delegates_through_cpu_pool(monkeypatch):
    """aembed_text routes embed_text through the dedicated pool and returns its vector."""
    from app import embeddings

    monkeypatch.setattr(embeddings, "_local_cpu_embed", lambda text: [0.5, 0.25])
    # Bypass the L1/L2 caches so the patched encoder path is exercised.
    monkeypatch.setattr(embeddings, "_load_cache", lambda key: None)
    monkeypatch.setattr(embeddings, "_save_cache", lambda key, vec: None)
    embeddings._embed_l1_cache._data.clear()

    vec = await embeddings.aembed_text("hello world")
    assert vec == [0.5, 0.25]


@pytest.mark.asyncio
async def test_aembed_multimodal_returns_text_vector(monkeypatch):
    """aembed_multimodal returns the multimodal payload built off the text embedding."""
    from app import embeddings

    monkeypatch.setattr(embeddings, "_local_cpu_embed", lambda text: [1.0])
    monkeypatch.setattr(embeddings, "_load_cache", lambda key: None)
    monkeypatch.setattr(embeddings, "_save_cache", lambda key, vec: None)
    embeddings._embed_l1_cache._data.clear()

    out = await embeddings.aembed_multimodal("desc", None)
    assert out["multimodal"] == [1.0]
    assert out["text"] == [1.0]
