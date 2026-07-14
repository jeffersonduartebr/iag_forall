# Objective: Tests for speculative (hedged) provider execution (perf #22).
"""Behavioral tests for services.hedged_execution.execute_with_hedge."""

from __future__ import annotations

import asyncio

import pytest
from app.services.hedged_execution import execute_with_hedge


@pytest.mark.asyncio
async def test_fast_primary_wins_without_launching_backup():
    """A primary that answers within the hedge delay wins alone; no backup is tried."""
    launched = []

    async def execute_fn(model):
        launched.append(model)
        if model == "primary":
            await asyncio.sleep(0.01)
            return ("primary-answer", {"m": model})
        await asyncio.sleep(1.0)  # backup would be slow, but should never run
        return ("backup-answer", {"m": model})

    result = await execute_with_hedge(
        ["primary", "backup"], execute_fn, hedge_delay_s=0.2, max_parallel=2
    )
    assert result.success is True
    assert result.model_used == "primary"
    assert result.models_tried == ["primary"]
    assert launched == ["primary"]


@pytest.mark.asyncio
async def test_slow_primary_lets_backup_win_and_cancels_primary():
    """A slow primary triggers the hedge; the faster backup wins and the primary is cancelled."""
    cancelled = {"primary": False}

    async def execute_fn(model):
        if model == "primary":
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                cancelled["primary"] = True
                raise
            return ("primary-answer", {})
        await asyncio.sleep(0.02)
        return ("backup-answer", {"m": model})

    result = await execute_with_hedge(
        ["primary", "backup"], execute_fn, hedge_delay_s=0.05, max_parallel=2
    )
    assert result.success is True
    assert result.model_used == "backup"
    assert set(result.models_tried) == {"primary", "backup"}
    # Give the event loop a tick to process the cancellation.
    await asyncio.sleep(0)
    assert cancelled["primary"] is True


@pytest.mark.asyncio
async def test_primary_failure_falls_through_to_backup():
    """When the primary raises, the backup is pulled forward and its success is returned."""

    async def execute_fn(model):
        if model == "primary":
            raise RuntimeError("primary down")
        return ("backup-answer", {"m": model})

    result = await execute_with_hedge(
        ["primary", "backup"], execute_fn, hedge_delay_s=0.2, max_parallel=2
    )
    assert result.success is True
    assert result.model_used == "backup"
    assert result.errors and result.errors[0]["model"] == "primary"


@pytest.mark.asyncio
async def test_all_candidates_failing_reports_failure_with_errors():
    """Exhausting all candidates yields an unsuccessful result carrying every error."""

    async def execute_fn(model):
        raise RuntimeError(f"{model} down")

    result = await execute_with_hedge(
        ["primary", "backup"], execute_fn, hedge_delay_s=0.05, max_parallel=2
    )
    assert result.success is False
    assert {e["model"] for e in result.errors} == {"primary", "backup"}


@pytest.mark.asyncio
async def test_empty_candidate_list_returns_failure():
    """No candidates is a graceful failure, not a crash."""

    async def execute_fn(model):  # pragma: no cover - never called
        return ("x", {})

    result = await execute_with_hedge([], execute_fn, hedge_delay_s=0.1)
    assert result.success is False
    assert result.model_used == ""


@pytest.mark.asyncio
async def test_duplicate_and_falsy_models_are_deduped():
    """Falsy/duplicate candidates collapse so the same model is not raced against itself."""
    launched = []

    async def execute_fn(model):
        launched.append(model)
        await asyncio.sleep(0.01)
        return ("ok", {"m": model})

    result = await execute_with_hedge(
        ["primary", "", "primary"], execute_fn, hedge_delay_s=0.2
    )
    assert result.success is True
    assert result.model_used == "primary"
    assert launched == ["primary"]
