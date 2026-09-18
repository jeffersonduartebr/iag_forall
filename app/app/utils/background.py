# -*- coding: utf-8 -*-
# Objective: Tracked fire-and-forget tasks for the asyncio event loop.
"""Spawn background coroutines safely.

``loop.create_task(coro)`` without keeping the returned task has three problems
on a request path: the loop holds only a weak reference (the task can be
garbage-collected mid-flight), exceptions surface only as "never retrieved"
warnings, and nothing bounds how many run at once.

:func:`spawn` keeps a strong reference until the task finishes, logs and
counts failures, and — when ``limit`` is given — drops new work for that
``name`` once ``limit`` tasks are in flight (counted in
``background_tasks_dropped_total``). :func:`drain` waits for pending tasks on
shutdown and cancels the stragglers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine, Dict, Optional, Set

logger = logging.getLogger(__name__)

_tasks: Set[asyncio.Task] = set()
_daemons: Set[asyncio.Task] = set()
_inflight: Dict[str, int] = {}


def _metrics():
    try:
        from app.observability import BACKGROUND_TASK_ERRORS, BACKGROUND_TASKS_DROPPED

        return BACKGROUND_TASKS_DROPPED, BACKGROUND_TASK_ERRORS
    except Exception:  # pragma: no cover - métricas nunca quebram o caminho quente
        return None, None


def _on_done(name: str, task: asyncio.Task) -> None:
    _tasks.discard(task)
    _daemons.discard(task)
    _inflight[name] = max(0, _inflight.get(name, 1) - 1)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("[background] task %s failed: %s", name, exc)
        _, errors = _metrics()
        if errors is not None:
            errors.labels(name=name).inc()


def spawn(
    coro: Coroutine[Any, Any, Any],
    *,
    name: str,
    limit: Optional[int] = None,
    daemon: bool = False,
) -> Optional[asyncio.Task]:
    """Schedule ``coro`` on the running loop and track it until completion.

    Returns ``None`` when the work was dropped because ``limit`` tasks named
    ``name`` are already running. Raises ``RuntimeError`` (after closing
    ``coro``) when there is no running loop, so callers can fall back to a
    synchronous path. ``daemon=True`` marks long-running loops that
    :func:`drain` cancels right away instead of waiting for.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        coro.close()
        raise
    if limit is not None and _inflight.get(name, 0) >= limit:
        coro.close()
        dropped, _ = _metrics()
        if dropped is not None:
            dropped.labels(name=name).inc()
        return None
    task = loop.create_task(coro, name=name)
    _tasks.add(task)
    if daemon:
        _daemons.add(task)
    _inflight[name] = _inflight.get(name, 0) + 1
    task.add_done_callback(lambda t: _on_done(name, t))
    return task


def pending() -> int:
    """Number of tracked tasks still running."""
    return len(_tasks)


async def drain(timeout: float = 5.0) -> None:
    """Cancel daemon loops, wait up to ``timeout`` s for the other tasks, then cancel the rest."""
    current = asyncio.current_task()
    daemons = [t for t in _daemons if t is not current and not t.done()]
    for task in daemons:
        task.cancel()
    if daemons:
        await asyncio.gather(*daemons, return_exceptions=True)
    waiting = [t for t in _tasks if t is not current and not t.done()]
    if not waiting:
        return
    _, still_pending = await asyncio.wait(waiting, timeout=timeout)
    for task in still_pending:
        task.cancel()
    if still_pending:
        logger.info("[background] cancelled %d task(s) on shutdown", len(still_pending))
        await asyncio.gather(*still_pending, return_exceptions=True)
