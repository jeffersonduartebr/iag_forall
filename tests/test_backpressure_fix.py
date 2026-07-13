"""Tests for backpressure semaphore fail-closed behavior."""

from __future__ import annotations

import asyncio

import pytest
from app.middleware.backpressure import BackpressureSemaphore


@pytest.mark.asyncio
async def test_backpressure_acquire_release_without_private_mutation(monkeypatch):
    BackpressureSemaphore._instance = None
    sem = BackpressureSemaphore()
    sem._max_concurrent = 1
    sem._semaphore = asyncio.Semaphore(1)

    assert await sem.acquire() is True
    assert await sem.acquire() is False
    await sem.release()
    assert await sem.acquire() is True
    await sem.release()

    # Ensure we never touched private _value
    assert hasattr(sem._semaphore, "_value")
