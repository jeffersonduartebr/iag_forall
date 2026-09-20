# Objective: One provider failing must not take the others down, and a 401 is not retryable.
"""The provider circuit breakers had two defects, and the second hid the first.

They never counted anything: ``@breaker`` on an ``async def`` uses pybreaker's
synchronous path, which gets back a coroutine object instead of an exception
and records a success. No provider circuit could open, whatever the provider
did.

And one instance was shared by OpenAI, OpenRouter, Anthropic and Gemini. With
the accounting fixed, that would mean one provider's outage opening the circuit
for three healthy ones.
"""

# Entra pela fachada, como a aplicação: importar `providers._infra` primeiro
# apanha-o a meio da inicialização.
import app.providers_async  # noqa: F401  (inicializa o pacote)
import httpx
import pybreaker
import pytest
from app.providers._breakers import CLOUD_BREAKERS, breaker_states, local_breaker, reset_breakers
from app.providers._infra import RETRYABLE_STATUS, _is_retryable
from app.utils.breaker_async import guarded_by


@pytest.fixture(autouse=True)
def closed_circuits():
    reset_breakers()
    yield
    reset_breakers()


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_the_four_cloud_providers_have_four_distinct_breakers():
    assert len({id(b) for b in CLOUD_BREAKERS.values()}) == 4


@pytest.mark.asyncio
async def test_opening_one_provider_leaves_the_others_closed():
    @guarded_by(CLOUD_BREAKERS["openai"])
    async def openai_down():
        raise RuntimeError("openai em baixo")

    for _ in range(CLOUD_BREAKERS["openai"].fail_max):
        with pytest.raises(Exception):
            await openai_down()

    assert CLOUD_BREAKERS["openai"].current_state == "open"
    for name in ("anthropic", "gemini", "openrouter"):
        assert CLOUD_BREAKERS[name].current_state == "closed", name
    assert local_breaker.current_state == "closed"


@pytest.mark.asyncio
async def test_a_healthy_provider_still_answers_while_another_is_open():
    @guarded_by(CLOUD_BREAKERS["openai"])
    async def openai_down():
        raise RuntimeError("openai em baixo")

    @guarded_by(CLOUD_BREAKERS["anthropic"])
    async def anthropic_ok():
        return "resposta"

    for _ in range(CLOUD_BREAKERS["openai"].fail_max):
        with pytest.raises(Exception):
            await openai_down()

    assert await anthropic_ok() == "resposta"


def test_breaker_states_reports_every_provider():
    states = breaker_states()
    assert set(states) == {"openai", "openrouter", "anthropic", "gemini", "local"}
    assert all(state == "closed" for state in states.values())


# ---------------------------------------------------------------------------
# One request, one failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_retry_sits_inside_the_breaker():
    """Otherwise one user request exhausts fail_max by itself.

    With the retry outside, tenacity calls the breaker once per attempt, so a
    single retryable failure counted five times against ``fail_max=5``.
    """
    import inspect

    from app.providers import _anthropic, _gemini, _openai

    for module in (_openai, _anthropic, _gemini):
        source = inspect.getsource(module)
        assert "@guarded_by(" in source, module.__name__
        # O decorador do breaker aparece antes do retry, logo fica por fora.
        assert source.index("@guarded_by(") < source.index("@COMMON_RETRY_STRATEGY"), module.__name__


# ---------------------------------------------------------------------------
# What is worth retrying
# ---------------------------------------------------------------------------


def http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://exemplo/api")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("erro", request=request, response=response)


@pytest.mark.parametrize("status", [401, 403, 404, 400, 422])
def test_a_client_error_is_not_retried(status):
    """An invalid API key was retried 5 times with backoff to 60 s — about two
    minutes per request — and would now also open the provider circuit. A
    configuration error must not look like an outage."""
    assert _is_retryable(http_error(status)) is False


@pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
def test_congestion_and_unavailability_are_retried(status):
    assert _is_retryable(http_error(status)) is True


def test_a_transport_fault_without_a_status_is_retried():
    """A connect timeout has no response at all; retrying is the whole point."""
    assert _is_retryable(httpx.ConnectTimeout("sem resposta")) is True


def test_an_unrelated_exception_is_never_retried():
    assert _is_retryable(ValueError("erro de programação")) is False


def test_a_circuit_breaker_error_is_not_retried():
    """Retrying an open circuit just burns the deadline."""
    assert _is_retryable(pybreaker.CircuitBreakerError("aberto")) is False
