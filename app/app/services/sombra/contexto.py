# Objective: Mark the current task as shadow execution so side-effecting paths stand down.
"""A shadow call must not touch policy state, provider health or system metrics (R2, R3, R10).

The provider layer checks :func:`em_sombra` and skips: circuit breakers, the provider-unavailable flag, the
throughput EMA (``tps:*``) and the provider cost/latency metrics. A ``ContextVar`` follows the asyncio task, so
real requests running concurrently in the same process are unaffected.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_EM_SOMBRA: ContextVar[bool] = ContextVar("aristo_em_sombra", default=False)


def em_sombra() -> bool:
    """Whether the running task is a shadow evaluation."""
    return _EM_SOMBRA.get()


@contextmanager
def modo_sombra() -> Iterator[None]:
    """Run the enclosed block (and the tasks it creates) in shadow mode."""
    token = _EM_SOMBRA.set(True)
    try:
        yield
    finally:
        _EM_SOMBRA.reset(token)
