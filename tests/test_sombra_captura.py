# Objective: Sampling and eligibility on the request path; the job carries the delivered inputs (R1, R2, R4, R9).
from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.services.sombra import captura
from sombra_fakes import ligar


def _ctx(**kw):
    base = dict(
        tenant_id="ifrn-caso1", query="Q", system_prompt="SYS", modality="text", image_b64=None, temperature=0.4,
        max_tokens=2048, response_format=None, tools=None, messages=None,
        candidates=["gemini/gemini-3.8-flash", "openrouter/deepseek/deepseek-v4.1-flash"],
        hints={"rag_filter": {"disciplina": "bd"}, "user_key": "P1", "episode_id": "E1", "workload_class": "reasoning"},
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _chamar(ctx, choice=None):
    choice = choice or SimpleNamespace(exploration_mode=False, decision={"bandit": {"explored": True}})
    outcome = SimpleNamespace(chosen="gemini/gemini-3.8-flash")
    result = {"answer": "A", "estimated_cost_usd": 0.002, "latency_s": 3.0, "metadata": {"prompt_tokens": 9}}
    bundle = {"context": "CTX", "grounded": True, "retrieval_mode": "full_retrieval"}
    return captura.talvez_agendar(ctx, choice, outcome, "PROMPT", bundle, result, 0.8)


@pytest.fixture
def fila(monkeypatch):
    import app.tasks as tasks

    enviados = []
    monkeypatch.setattr(tasks.task_shadow_evaluate, "delay", lambda job: enviados.append(job))
    monkeypatch.setattr("app.correlation.get_correlation_id", lambda: "req-xyz")
    monkeypatch.setattr(captura, "em_aquecimento", lambda: False)  # estes testes são do campo
    return enviados


def test_the_shadow_is_paused_during_the_warmup(monkeypatch, fila):
    ligar(monkeypatch)
    monkeypatch.setattr(captura, "em_aquecimento", lambda: True)
    assert _chamar(_ctx()) is False and fila == []


def test_the_warmup_flag_follows_the_regime(monkeypatch):
    from app.services.regime import config as regime

    for ativo in (True, False):
        monkeypatch.setattr(regime, "carregar", lambda a=ativo: SimpleNamespace(em_aquecimento=lambda: a))
        assert captura.em_aquecimento() is ativo


def test_the_draw_is_deterministic_and_matches_the_rate():
    ids = [f"r{i}" for i in range(4000)]
    taxa = sum(captura.sorteada(i, 0.15) for i in ids) / len(ids)
    assert abs(taxa - 0.15) < 0.02 and [captura.sorteada(i, 0.15) for i in ids[:50]] == [captura.sorteada(i, 0.15) for i in ids[:50]]
    assert not captura.sorteada("x", 0.0) and captura.sorteada("x", 1.0)


def test_a_sampled_request_is_enqueued_with_the_delivered_inputs(monkeypatch, fila):
    ligar(monkeypatch)
    assert _chamar(_ctx())
    [job] = fila
    assert (job["prompt_final"], job["system_prompt"], job["contexto"], job["temperatura"], job["max_tokens"]) == (
        "PROMPT", "SYS", "CTX", 0.4, 2048)
    assert job["candidatas"] == ["openrouter/deepseek/deepseek-v4.1-flash"] and job["regime_entrega"] == "exploracao"
    assert job["estrato"] == {"disciplina": "bd", "faixa_incerteza": "alta", "modalidade": "text"}
    assert (job["participante"], job["episode_id"], job["caso"], job["p_nominal"]) == ("P1", "E1", 1, 1.0)


@pytest.mark.parametrize("mudanca", [
    {"tenant_id": "outro"}, {"tools": [{"type": "function"}]}, {"messages": [{"role": "user"}]}, {"candidates": []},
])
def test_ineligible_requests_are_not_sampled(monkeypatch, fila, mudanca):
    ligar(monkeypatch)
    assert not _chamar(_ctx(**mudanca)) and fila == []


def test_pinned_instrument_calls_and_the_switch(monkeypatch, fila):
    ligar(monkeypatch)
    assert not _chamar(_ctx(hints={"pinned_model": "gemini/x"}))
    ligar(monkeypatch, ligada=False)
    assert not _chamar(_ctx()) and fila == []


def test_an_enqueue_failure_never_reaches_the_request(monkeypatch, fila):
    import app.tasks as tasks

    ligar(monkeypatch)
    monkeypatch.setattr(tasks.task_shadow_evaluate, "delay", lambda job: (_ for _ in ()).throw(ConnectionError("broker")))
    assert _chamar(_ctx()) is False


@pytest.mark.asyncio
async def test_the_routing_path_enqueues_without_changing_the_answer(monkeypatch, fila):
    from app.services.router_execution import route_and_answer_internal_impl
    from router_fakes import deps_for_execution

    async def _responder(**kwargs):
        return "resposta", {"prompt_tokens": 1, "completion_tokens": 1, "cost_per_1k": 0.1}

    def _rodar():
        deps = deps_for_execution()
        deps.update(check_cache=lambda *a, **k: None, call_model=_responder, _safe_setting_bool=lambda k, d=False: False)
        return route_and_answer_internal_impl(
            deps=deps, query="pergunta", system_prompt="SYS", use_rag=False, max_tokens=64, temperature=0.3,
            modality="text", image_b64=None, rag_modality="text", use_cache=False, tenant_id="ifrn-caso1",
        )

    ligar(monkeypatch, ligada=False)
    sem = await _rodar()
    ligar(monkeypatch)
    com = await _rodar()
    assert com["answer"] == sem["answer"] == "resposta" and com["model"] == sem["model"]
    assert len(fila) == 1 and fila[0]["prompt_final"] and fila[0]["modelo_entregue"] == com["model"]
