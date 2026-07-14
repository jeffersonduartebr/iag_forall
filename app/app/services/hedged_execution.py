# -*- coding: utf-8 -*-
# Objective: Speculative (hedged) provider execution to cut tail latency (perf #22).
"""Hedged request execution — the tied-request pattern from *The Tail at Scale*.

Instead of waiting for a slow primary provider to time out before trying a
backup (the sequential fallback chain), fire the primary and, if it has not
answered within ``hedge_delay_s``, launch a backup **in parallel** and take
whichever finishes first — cancelling the losers.

This trades extra provider cost (up to ``max_parallel`` concurrent calls on
the hedged fraction of traffic) for a shorter p95/p99. It is disabled by
default (``REQUEST_HEDGING_ENABLED=0``) and only engaged when a distinct
backup candidate exists.

The return type mirrors :class:`reliability.FallbackResult` so the router
consumes a hedged outcome exactly like a fallback-chain outcome.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List

from ..error_handling import ErrorCategory, log_provider_error
from ..reliability import FallbackResult

logger = logging.getLogger(__name__)


def _dedupe_preserving_order(models: List[str]) -> List[str]:
    """Return models without falsy/duplicate entries, keeping first-seen order."""
    seen: set[str] = set()
    ordered: List[str] = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


async def execute_with_hedge(
    models: List[str],
    execute_fn: Callable[[str], Awaitable[Any]],
    hedge_delay_s: float,
    max_parallel: int = 2,
) -> FallbackResult:
    """Race provider calls with staggered starts, returning the first success.

    Args:
        models: Ordered candidates; ``models[0]`` is the primary.
        execute_fn: Async callable taking a model name and returning its result.
        hedge_delay_s: Seconds to wait on the in-flight set before launching the
            next backup. Also the polling cadence while candidates are exhausted.
        max_parallel: Maximum number of concurrent in-flight provider calls.

    Returns:
        A :class:`FallbackResult`; ``model_used`` is the model that won the race.
    """
    ordered = _dedupe_preserving_order(models)
    if not ordered:
        return FallbackResult(
            success=False,
            result=None,
            model_used="",
            models_tried=[],
            errors=[{"model": "", "error": "no candidates", "category": ErrorCategory.PROVIDER_UNAVAILABLE.value}],
        )

    cap = max(1, int(max_parallel))
    delay = max(0.05, float(hedge_delay_s))
    tasks: Dict[asyncio.Task, str] = {}
    models_tried: List[str] = []
    errors: List[Dict[str, Any]] = []
    next_idx = 0

    def _launch_next() -> None:
        nonlocal next_idx
        model = ordered[next_idx]
        next_idx += 1
        models_tried.append(model)
        tasks[asyncio.ensure_future(execute_fn(model))] = model

    async def _cancel_pending() -> None:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        tasks.clear()

    _launch_next()  # primary
    while tasks:
        done, _pending = await asyncio.wait(
            set(tasks), timeout=delay, return_when=asyncio.FIRST_COMPLETED
        )
        if not done:
            # Primary (or current set) is slow: stagger in a backup if we can.
            if next_idx < len(ordered) and len(tasks) < cap:
                _launch_next()
            continue

        for task in done:
            model = tasks.pop(task)
            try:
                result = task.result()
            except asyncio.CancelledError:  # pragma: no cover - defensive
                continue
            except Exception as exc:
                info = log_provider_error(exc, model, operation="hedge_execute")
                errors.append({"model": model, "error": str(exc), "category": info.category.value})
                continue
            # First success wins the race — cancel every straggler.
            await _cancel_pending()
            if len(models_tried) > 1:
                logger.info("[Hedge] %s won over %d candidate(s)", model, len(models_tried))
            return FallbackResult(
                success=True,
                result=result,
                model_used=model,
                models_tried=models_tried,
                errors=errors,
            )

        # A slot freed up (failure): pull the next backup forward immediately.
        if next_idx < len(ordered) and len(tasks) < cap:
            _launch_next()

    return FallbackResult(
        success=False,
        result=None,
        model_used=models_tried[-1] if models_tried else ordered[0],
        models_tried=models_tried,
        errors=errors,
    )
