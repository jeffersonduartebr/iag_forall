# Objective: Rate-limit decision tables (preempt to async, backpressure state, quota, cleanup) — pure logic.
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from app.middleware import rate_limit as rl

CONGESTED = {"current_limit": 1, "total_inflight": 0, "max_queue_wait_ms": 0.0, "utilization": 0.0, "pressure_state": "congested"}
ELEVATED = dict(CONGESTED, pressure_state="elevated")


@pytest.mark.parametrize(
    ("snapshot", "state", "expected"),
    [
        ({}, "normal", False),
        ({"current_limit": 4, "total_inflight": 1, "max_queue_wait_ms": 110.0}, "elevated", True),
        ({"current_limit": 4, "total_inflight": 1, "utilization": 0.9}, "congested", True),
        ({"current_limit": 4, "total_inflight": 1, "utilization": 0.1}, "congested", False),
        ({"current_limit": 4, "total_inflight": 2}, "elevated", True),
    ],
)
def test_preempt_to_async_decision(snapshot, state, expected):
    cfg = {"sync_queue_wait_ms": 250.0, "elevated_utilization": 0.8}
    assert rl.RateLimitMiddleware(app=None)._should_preempt_to_async(snapshot, state, cfg) is expected


@pytest.mark.parametrize(("bp", "expected"), [(0.96, "congested"), (0.9, "elevated"), (0.1, "normal")])
def test_backpressure_utilization_drives_candidate_state(monkeypatch, bp, expected):
    monkeypatch.setattr(rl, "get_backpressure", lambda: SimpleNamespace(get_stats=lambda: {"utilization": bp}))
    cfg = {"congested_utilization": 1.0, "congested_queue_wait_ms": 1000.0, "elevated_utilization": 0.8,
           "elevated_queue_wait_ms": 500.0}
    assert rl.RateLimitMiddleware(app=None)._candidate_pressure_state({}, cfg) == expected


def test_quota_is_none_when_healthy():
    assert rl.RateLimitMiddleware(app=None)._quota_for("admin_eval_governance", "normal", {}, {}) is None


@pytest.mark.asyncio
async def test_periodic_cleanup_survives_cleanup_errors(monkeypatch):
    calls = {"sleep": 0, "cleanup": 0}

    async def _sleep(_s):
        calls["sleep"] += 1
        if calls["sleep"] > 2:
            raise asyncio.CancelledError

    async def _cleanup():
        calls["cleanup"] += 1
        if calls["cleanup"] == 1:
            raise RuntimeError("redis gone")
        return 3

    monkeypatch.setattr(rl.asyncio, "sleep", _sleep)
    monkeypatch.setattr(rl.rate_limit_store, "cleanup", _cleanup)
    with pytest.raises(asyncio.CancelledError):
        await rl.periodic_cleanup()
    assert calls == {"sleep": 3, "cleanup": 2}
