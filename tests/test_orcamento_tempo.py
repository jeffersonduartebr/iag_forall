# Objective: Time budgets follow each model's measured throughput and always leave room for a fallback.
"""Run 2 of the Caso 1 load test: fixed 20/25 s budgets timed out every long answer from the T4, and a
chain of one model ("All 1 models failed") turned each timeout into a 502. DeepSeek spent ``max_tokens``
reasoning and came back empty, which the router turned into an abstention."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.services import orcamento_tempo as ot
from app.services import router_provider_stage as ps


def test_throughput_ema_is_per_model_and_ignores_the_route_prefix(fake_redis):
    assert ot.vazao("ollama/granite") == 10.0 and ot.vazao("openrouter/x/y") == 40.0  # padrões
    ot.registrar_vazao("granite", 200, 10.0)  # o provedor vê o nome sem a rota
    assert ot.vazao("ollama/granite") == 20.0
    ot.registrar_vazao("granite", 100, 10.0)
    assert ot.vazao("ollama/granite") == pytest.approx(0.7 * 20 + 0.3 * 10)
    ot.registrar_vazao("granite", 5, 0.1)  # resposta curta mede a latência inicial, não a vazão
    assert ot.vazao("ollama/granite") == pytest.approx(17.0)


def test_request_deadline_grows_with_tokens_and_is_capped():
    curto, longo = ot.prazo_da_requisicao(256, 25), ot.prazo_da_requisicao(4096, 25)
    assert 25 < curto < longo <= 420.0
    assert ot.prazo_da_requisicao(256, 239) == 239  # nunca abaixo do piso do workload


def test_call_timeout_reserves_time_for_the_fallback_while_it_can(fake_redis):
    assert ot.prazo_da_chamada("ollama/g", 2048, restante=200) == 200 - 45  # local lento: limitado pela reserva
    assert ot.prazo_da_chamada("ollama/g", 2048, restante=60) == 59.5  # sem espaço para outro: usa tudo
    assert ot.prazo_da_chamada("ollama/g", 10, restante=200) == pytest.approx(4 + 1.3 * 10 / 10)
    assert ot.prazo_da_chamada("ollama/g", 10, restante=200, piso=20) == 20


def test_reasoning_quota_is_added_only_to_models_that_reason(monkeypatch):
    assert ot.tokens_totais(512) == 512 + 4096 and ot.tokens_totais(512, com_raciocinio=False) == 512
    monkeypatch.setattr(ot.settings, "get", lambda k, d=None: 0 if k == "REASONING_BUDGET_TOKENS" else d)
    assert ot.tokens_totais(512) == 512  # zero desliga, não volta ao padrão


def test_fallback_order_puts_cloud_models_first():
    pool = ["ollama/a", "gemini/g", "ollama/b", "openrouter/d", "gemini/g"]
    assert ot.ordem_de_fallback(pool, "ollama/a") == ["gemini/g", "openrouter/d", "ollama/b"]


def _ctx(deps, **hints):
    return SimpleNamespace(
        deps=deps, runtime_hints=hints, max_tokens=64, modality="text", image_b64=None, temperature=0.1,
        hints=hints, tools=None, tool_choice=None, messages=None, response_format=None, system_prompt="",
    )


class _Error(Exception):
    def __init__(self, model, message, category, retryable):
        super().__init__(message)
        self.category = category


@pytest.mark.asyncio
@pytest.mark.parametrize(("retry_empty", "levanta"), [(True, True), (False, False)])
async def test_an_empty_answer_fails_the_attempt_only_when_another_model_can_answer(retry_empty, levanta):
    async def call_model(**kwargs):
        return "  ", {"finish_reason": "length"}

    execute = ps._provider_call(_ctx({"call_model": call_model, "ProviderCallError": _Error}), "p", retry_empty)
    if levanta:
        with pytest.raises(_Error) as exc:
            await execute("openrouter/deepseek")
        assert exc.value.category == "empty_answer"
    else:
        assert await execute("openrouter/deepseek") == ("  ", {"finish_reason": "length"})


@pytest.mark.asyncio
async def test_a_tool_call_turn_is_not_an_empty_answer():
    async def call_model(**kwargs):
        return "", {"tool_calls": [{"id": "1"}]}

    execute = ps._provider_call(_ctx({"call_model": call_model, "ProviderCallError": _Error}), "p", True)
    assert (await execute("m"))[1]["tool_calls"]


@pytest.mark.asyncio
async def test_the_registry_chain_is_replaced_by_the_callers_candidates(monkeypatch):
    from app import reliability

    tentados = []

    async def execute(model):
        tentados.append(model)
        if model != "gemini/g":
            raise RuntimeError("down")
        return "ok", {}

    monkeypatch.setattr(reliability, "call_through_breaker", lambda _b, fn, m: fn(m))
    out = await reliability.execute_with_fallback("ollama/a", execute, max_fallbacks=2, candidates=["gemini/g"])
    assert out.success and tentados == ["ollama/a", "gemini/g"]
