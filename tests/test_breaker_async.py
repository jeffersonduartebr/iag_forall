# Objective: The async circuit-breaker call must actually work and actually account.
"""``pybreaker.CircuitBreaker.call_async`` is Tornado-only and raises NameError.

Tornado is not a dependency of this project, so every call through the library's
own async helper failed with ``NameError: name 'gen' is not defined``. Inside
the fallback loop that surfaced as "this model failed", so the chain recorded
every candidate as broken and gave up — it could never have succeeded, whichever
model was healthy.

These tests pin both halves of the replacement: that it runs at all, and that it
keeps the breaker's accounting, because a breaker that never trips is as useless
as one that never runs.
"""

import asyncio

import pybreaker
import pytest
from app.utils.breaker_async import call_through_breaker


def a_breaker(fail_max=2, reset_timeout=60):
    return pybreaker.CircuitBreaker(fail_max=fail_max, reset_timeout=reset_timeout, name="teste")


# ---------------------------------------------------------------------------
# The bug itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_librarys_own_helper_is_unusable():
    """Pins *why* this module exists; delete it only when pybreaker is fixed."""

    async def _ok(model):
        return f"ok-{model}"

    with pytest.raises(NameError, match="gen"):
        await a_breaker().call_async(_ok, "m")


@pytest.mark.asyncio
async def test_a_successful_call_returns_its_value():
    async def _ok(model):
        return f"ok-{model}"

    assert await call_through_breaker(a_breaker(), _ok, "m") == "ok-m"


@pytest.mark.asyncio
async def test_the_original_exception_reaches_the_caller():
    """The fallback loop classifies the error, so it must be the real one."""

    async def _boom(model):
        raise ValueError("falha do provider")

    with pytest.raises(ValueError, match="falha do provider"):
        await call_through_breaker(a_breaker(), _boom, "m")


# ---------------------------------------------------------------------------
# The accounting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_failures_open_the_circuit():
    breaker = a_breaker(fail_max=2)

    async def _boom(model):
        raise ValueError("falha")

    # Abaixo do limite, a excepção do provider chega intacta ao chamador.
    with pytest.raises(ValueError):
        await call_through_breaker(breaker, _boom, "m")
    assert breaker.current_state == "closed"

    # Na chamada que faz disparar, o pybreaker substitui a excepção original
    # por CircuitBreakerError. É o mesmo que o `breaker.call` síncrono faz, e
    # `execute_with_fallback` já classifica esse caso como CIRCUIT_OPEN — mas
    # significa que o erro real do provider se perde exactamente nessa chamada.
    with pytest.raises(pybreaker.CircuitBreakerError):
        await call_through_breaker(breaker, _boom, "m")
    assert breaker.current_state == "open"


@pytest.mark.asyncio
async def test_an_open_circuit_does_not_call_the_function():
    breaker = a_breaker(fail_max=1)
    calls = []

    async def _boom(model):
        calls.append(model)
        raise ValueError("falha")

    # fail_max=1: esta chamada falha e faz disparar de uma vez.
    with pytest.raises(pybreaker.CircuitBreakerError):
        await call_through_breaker(breaker, _boom, "m")
    assert breaker.current_state == "open"

    with pytest.raises(pybreaker.CircuitBreakerError):
        await call_through_breaker(breaker, _boom, "m")
    assert calls == ["m"], "a função foi chamada com o circuito aberto"


@pytest.mark.asyncio
async def test_a_success_resets_the_failure_counter():
    breaker = a_breaker(fail_max=3)

    async def _boom(model):
        raise ValueError("falha")

    async def _ok(model):
        return "ok"

    with pytest.raises(ValueError):
        await call_through_breaker(breaker, _boom, "m")
    assert breaker.fail_counter == 1

    await call_through_breaker(breaker, _ok, "m")
    assert breaker.fail_counter == 0


@pytest.mark.asyncio
async def test_an_excluded_exception_is_not_counted_as_a_failure():
    """A 400 from a provider says nothing about the provider's health."""

    class ClientRequestError(Exception):
        pass

    breaker = a_breaker(fail_max=1)
    breaker.add_excluded_exception(ClientRequestError)

    async def _client_error(model):
        raise ClientRequestError("pedido inválido")

    with pytest.raises(ClientRequestError):
        await call_through_breaker(breaker, _client_error, "m")
    assert breaker.current_state == "closed"


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_calls_are_not_serialised_by_the_lock():
    """The breaker's own lock must not be held across the await.

    pybreaker guards its state with a re-entrant *thread* lock. Held across an
    await it would give no mutual exclusion at all between asyncio tasks (same
    thread, so it is reentrant) while blocking any real worker thread for the
    whole duration of a remote call. It is taken only around the transitions.
    """
    breaker = a_breaker()
    order = []

    async def _slow(model):
        order.append(f"inicio-{model}")
        await asyncio.sleep(0.02)
        order.append(f"fim-{model}")
        return model

    await asyncio.gather(
        call_through_breaker(breaker, _slow, "a"),
        call_through_breaker(breaker, _slow, "b"),
    )

    # Se o lock fosse mantido durante o await, "a" terminaria antes de "b"
    # começar e a ordem seria início-a, fim-a, início-b, fim-b.
    assert order[:2] == ["inicio-a", "inicio-b"], order


# ---------------------------------------------------------------------------
# The decorator, which is where the provider layer was broken
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pybreakers_own_decorator_never_sees_an_async_failure():
    """Pins the defect: this is why no provider circuit could ever open.

    ``@breaker`` wraps the function with the *synchronous* ``call``. Applied to
    an ``async def``, that returns the coroutine object instead of raising, so
    the breaker records a success every time.
    """
    breaker = a_breaker(fail_max=2)

    @breaker
    async def _boom():
        raise RuntimeError("provider em baixo")

    for _ in range(5):
        with pytest.raises(RuntimeError):
            await _boom()

    assert breaker.fail_counter == 0
    assert breaker.current_state == "closed"


@pytest.mark.asyncio
async def test_guarded_by_counts_the_failure():
    from app.utils.breaker_async import guarded_by

    breaker = a_breaker(fail_max=3)

    @guarded_by(breaker)
    async def _boom():
        raise RuntimeError("provider em baixo")

    with pytest.raises(RuntimeError):
        await _boom()
    assert breaker.fail_counter == 1


@pytest.mark.asyncio
async def test_guarded_by_preserves_the_function_identity():
    """The provider methods are looked up by name in tests and logs."""
    from app.utils.breaker_async import guarded_by

    @guarded_by(a_breaker())
    async def generate(self, prompt):
        return prompt

    assert generate.__name__ == "generate"
