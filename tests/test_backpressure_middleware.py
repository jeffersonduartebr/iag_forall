# Objective: Test coverage for backpressure middleware behavior and regressions.
"""Unit tests for the global backpressure semaphore and middleware helpers."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_backpressure_semaphore_respects_capacity(monkeypatch):
    """The semaphore should reject requests immediately once capacity is exhausted."""
    from app.middleware import backpressure as bp

    bp.BackpressureSemaphore._instance = None
    monkeypatch.setattr(bp.settings, "get", lambda key, default=None: 2 if key == "MAX_CONCURRENT_REQUESTS" else default)

    semaphore = bp.BackpressureSemaphore()
    assert await semaphore.acquire() is True
    assert await semaphore.acquire() is True
    assert await semaphore.acquire() is False

    await semaphore.release()
    assert semaphore.current_load == 1
    assert await semaphore.acquire() is True


@pytest.mark.asyncio
async def test_backpressure_release_tolerates_extra_release(monkeypatch):
    """Releasing more times than acquired should not crash or underflow counters."""
    from app.middleware import backpressure as bp

    bp.BackpressureSemaphore._instance = None
    monkeypatch.setattr(bp.settings, "get", lambda key, default=None: 1 if key == "MAX_CONCURRENT_REQUESTS" else default)

    semaphore = bp.BackpressureSemaphore()
    await semaphore.release()
    assert semaphore.current_load == 0
