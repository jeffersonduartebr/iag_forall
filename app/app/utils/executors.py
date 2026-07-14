# -*- coding: utf-8 -*-
# Objective: Dedicated CPU-bound thread pool isolated from the I/O executor (perf #23).
"""A dedicated thread pool for CPU-bound hot-path work (embeddings, rerank).

Local embedding (``SentenceTransformer.encode``) is CPU-bound and holds the
GIL. Running it on the default asyncio executor — the same pool that services
blocking I/O such as DB reads via ``asyncio.to_thread`` — lets a burst of
embeddings starve I/O work (head-of-line blocking).

Routing embeddings through this separate, small, bounded pool isolates that
contention. The pool is intentionally small: because encode holds the GIL,
extra threads do not add CPU throughput, they only serialize on the GIL; the
goal here is *isolation from I/O*, not CPU parallelism.

Sizing: ``EMBED_CPU_THREADS`` env override, else ``clamp(2..4, cpu_count)``.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_cpu_executor: Optional[ThreadPoolExecutor] = None
_lock = threading.Lock()


def _resolve_workers() -> int:
    """Resolve the CPU pool size from env override or a clamped CPU-count default."""
    raw = os.getenv("EMBED_CPU_THREADS", "").strip()
    if raw:
        try:
            parsed = int(raw)
            if parsed > 0:
                return parsed
        except ValueError:
            logger.warning("[executors] Invalid EMBED_CPU_THREADS=%r; using default", raw)
    cpu = os.cpu_count() or 4
    return max(2, min(4, cpu))


def get_cpu_executor() -> ThreadPoolExecutor:
    """Return the process-wide CPU-bound executor, creating it on first use."""
    global _cpu_executor
    if _cpu_executor is None:
        with _lock:
            if _cpu_executor is None:
                workers = _resolve_workers()
                _cpu_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cpu-embed")
                logger.info("[executors] CPU-bound pool created: workers=%d", workers)
    return _cpu_executor


async def run_cpu_bound(fn: Callable[..., _T], *args: Any) -> _T:
    """Run a blocking CPU-bound callable on the dedicated pool, awaiting its result."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_cpu_executor(), functools.partial(fn, *args))


def shutdown_cpu_executor() -> None:
    """Tear down the CPU pool (used on app shutdown / test isolation)."""
    global _cpu_executor
    if _cpu_executor is not None:
        _cpu_executor.shutdown(wait=False)
        _cpu_executor = None
