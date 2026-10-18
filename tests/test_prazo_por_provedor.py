# Objective: A degraded cloud provider yields to the next candidate instead of holding the request to the deadline.
"""Rehearsal of 2026-10-18: Vertex ``gemini-3.8-flash`` took 52 s to answer "diga ok". The router waited for the
global deadline (248 s) and answered 504, for three compounding reasons:

- the Gemini adapter passed no timeout to its SDK, so the call only ended when the deadline cancelled the ``await``;
- a cloud call's budget was sized by throughput alone (answer + reasoning quota), about 200 s;
- a read timeout was retried on the *same* model by the provider's retry strategy.

Each test below fails without its fix.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import app.providers_async  # noqa: F401  (carrega os provedores na ordem certa)
import httpx
import pytest
from app.providers import _gemini
from app.providers._infra import CLOUD_BREAKERS, _is_retryable
from app.reliability import execute_with_fallback, get_circuit_breaker_manager
from app.router_strategy import _get_circuit_breaker_penalty
from app.services import orcamento_tempo as ot
from app.services import router_provider_stage as ps


def test_a_cloud_call_is_capped_so_a_fallback_still_fits(fake_redis):
    assert ot.prazo_da_chamada("gemini/gemini-3.8-flash", 2048, restante=248) == 60.0
    assert ot.prazo_da_chamada("gemini/gemini-3.8-flash", 2048, restante=248, piso=90) == 60.0  # o teto vence o piso


def test_the_cap_leaves_local_models_alone_and_zero_disables_it(fake_redis, monkeypatch):
    assert ot.prazo_da_chamada("ollama/g", 2048, restante=400) == pytest.approx(4 + 1.3 * 2048 / 10)
    monkeypatch.setattr(ot.settings, "get", lambda k, d=None: 0 if k == "PRAZO_CHAMADA_NUVEM_S" else d)
    assert ot.prazo_da_chamada("gemini/g", 2048, restante=400) > 60


def test_the_router_passes_the_capped_timeout_to_the_provider(fake_redis):
    hints = {"request_deadline_ts": time.monotonic() + 248, "provider_timeout_seconds": 0}
    ctx = SimpleNamespace(runtime_hints=hints, hints={}, max_tokens=2048)
    assert ps._call_timeout(ctx, "gemini/gemini-3.8-flash") <= 60.0


def test_a_pinned_instrument_model_keeps_its_full_budget(fake_redis):
    """``pinned_model`` (adherence measurement) runs with no fallback: capping it would only cut a slow answer."""
    hints = {"request_deadline_ts": time.monotonic() + 248, "provider_timeout_seconds": 0}
    ctx = SimpleNamespace(runtime_hints=hints, hints={"max_fallbacks": 0}, max_tokens=2048)
    assert ps._call_timeout(ctx, "gemini/gemini-3.8-flash") > 60.0


def test_a_read_timeout_is_not_retried_on_the_same_model():
    """The fallback chain handles an exhausted wait, with another candidate; a quick connect fault still retries."""
    assert _is_retryable(httpx.ReadTimeout("lento")) is False
    assert _is_retryable(httpx.ConnectTimeout("sem resposta")) is True
    from openai import APITimeoutError

    assert _is_retryable(APITimeoutError(request=httpx.Request("POST", "https://x"))) is False


class _SdkLento:
    """Fake ``google-genai`` whose generation blocks until released, recording the config it got."""

    def __init__(self):
        self.liberar, self.configs = threading.Event(), []

    def Client(self, **kwargs):  # noqa: N802 (espelha a API do SDK)
        def _gerar(**kw):
            self.configs.append(kw["config"])
            self.liberar.wait(5)
            return SimpleNamespace(text="tarde demais", usage_metadata=None)

        return SimpleNamespace(models=SimpleNamespace(generate_content=_gerar))


@pytest.fixture
def sdk_lento(monkeypatch):
    fake = _SdkLento()
    monkeypatch.setattr(_gemini, "google_genai", fake)
    monkeypatch.setattr(_gemini, "GEMINI_VERTEX_PROJECT", "p")
    _gemini._CLIENTES.clear()
    yield fake
    fake.liberar.set()
    _gemini._CLIENTES.clear()
    CLOUD_BREAKERS["gemini"].close()


def test_gemini_hands_the_timeout_to_its_sdk(sdk_lento):
    sdk_lento.liberar.set()
    options = _gemini._GenOptions(0.2, 100, None, None, None, None, 35.0)
    _gemini.GeminiProvider.GeminiAdapter._generate_genai("gemini-3.8-flash", [{"text": "oi"}], options)
    assert sdk_lento.configs[-1]["http_options"] == {"timeout": 35000}


@pytest.mark.asyncio
async def test_a_hung_gemini_call_ends_at_its_timeout_and_is_tried_once(sdk_lento, monkeypatch):
    monkeypatch.setattr(_gemini, "MARGEM_PRAZO_S", 0.0)
    inicio = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        await _gemini.GeminiProvider().generate("diga ok", model="gemini-3.8-flash", timeout_seconds=1)
    assert time.monotonic() - inicio < 3, "esperou além do prazo da chamada"
    assert len(sdk_lento.configs) == 1, "a espera esgotada foi repetida no mesmo modelo"


@pytest.mark.asyncio
async def test_a_timed_out_model_yields_to_the_next_candidate_and_is_penalised():
    lento, saudavel = "gemini/ensaio-degradado", "openrouter/ensaio-saudavel"
    chamados = []

    async def _executar(modelo):
        chamados.append(modelo)
        if modelo == lento:
            raise asyncio.TimeoutError()
        return "ok", {}

    resultado = await execute_with_fallback(lento, _executar, max_fallbacks=2, candidates=[saudavel, lento])
    assert resultado.success and resultado.model_used == saudavel
    assert chamados == [lento, saudavel], "o modelo que estourou o prazo foi repetido na mesma requisição"
    assert resultado.errors[0]["category"] == "provider_timeout"
    # A política já pune falhas pelo disjuntor do modelo: o escore da próxima escolha sai multiplicado por < 1.
    assert get_circuit_breaker_manager().get_status(lento)["fail_counter"] == 1
    assert _get_circuit_breaker_penalty(lento) < 1.0
    get_circuit_breaker_manager().reset_breaker(lento)
