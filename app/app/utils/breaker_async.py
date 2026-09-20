# Objective: Run an awaitable under a pybreaker circuit breaker, without Tornado.
"""Native-asyncio replacement for ``pybreaker.CircuitBreaker.call_async``.

``call_async`` in pybreaker 1.0.1 is implemented with ``tornado.gen.coroutine``.
Tornado is not a dependency of this project, so the import at the top of
pybreaker fails, ``HAS_TORNADO_SUPPORT`` becomes ``False`` — and ``call_async``
still refers to ``gen``. Every call through it therefore raises
``NameError: name 'gen' is not defined``.

That error surfaced as "this model failed" inside the fallback loop, so the
chain recorded every candidate as broken and gave up. The fallback path could
never have succeeded, no matter which model was healthy; it was invisible only
because the flag that reaches it was off and the tests substituted
``execute_with_fallback`` wholesale.

This mirrors ``CircuitBreakerState.call`` — the synchronous path pybreaker does
implement — with one deliberate difference: the breaker's ``threading.RLock``
is **not** held across the ``await``. Holding it would be worse than useless
here. It is reentrant per *thread*, so several asyncio tasks on the same event
loop would all acquire it and see no mutual exclusion at all, while any real
worker thread would block for the whole duration of a remote call. The lock is
taken only around the state transitions, which are the parts that actually race.
"""

from __future__ import annotations

import threading
from typing import Any, Awaitable, Callable, TypeVar

import pybreaker

T = TypeVar("T")

#: pybreaker guards its state with a re-entrant lock; the transitions below are
#: short and non-blocking, so one module-level lock around them is enough.
_TRANSITION_LOCK = threading.Lock()


async def call_through_breaker(
    breaker: pybreaker.CircuitBreaker,
    func: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Await ``func(*args, **kwargs)`` under ``breaker``'s accounting.

    Raises :class:`pybreaker.CircuitBreakerError` when the circuit is open, and
    re-raises whatever ``func`` raised otherwise — the same contract as the
    synchronous ``breaker.call``.
    """
    with _TRANSITION_LOCK:
        state = breaker.state
        # Raises CircuitBreakerError when the circuit is open; this is the
        # check that makes an open breaker skip the call entirely.
        state.before_call(func, *args, **kwargs)
        listeners = list(breaker.listeners)
    for listener in listeners:
        listener.before_call(breaker, func, *args, **kwargs)

    try:
        result = await func(*args, **kwargs)
    except BaseException as exc:
        with _TRANSITION_LOCK:
            # ``_handle_error`` counts the failure, notifies listeners, may trip
            # the circuit, and re-raises. An exception the breaker is configured
            # to exclude is counted as a success instead, which is why this is
            # delegated rather than reimplemented.
            breaker.state._handle_error(exc)
        raise  # pragma: no cover - _handle_error always re-raises

    with _TRANSITION_LOCK:
        breaker.state._handle_success()
    return result
