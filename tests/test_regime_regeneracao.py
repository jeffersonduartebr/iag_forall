# Objective: An exploratory answer that fails the uncertainty check is regenerated with the exploitation config.
from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.services.regime import regeneracao
from app.services.router_provider_stage import ProviderOutcome
from app.services.router_stages import RouteChoice


def _escolha(explorou=True):
    return RouteChoice(
        chosen="explorado",
        top2=["explorado"],
        decision={"regime": {"explorou": explorou, "aproveitamento": "guloso", "regenerado": False}},
    )


def _ctx():
    return SimpleNamespace(hints={"workload_class": "knowledge_lookup", "detected_complexity": "moderate"})


@pytest.fixture
def chamadas(monkeypatch):
    feitas = []

    async def _executar(ctx, choice, prompt):
        feitas.append(choice.chosen)
        return ProviderOutcome("resposta do guloso", {}, choice.chosen)

    monkeypatch.setattr(regeneracao, "execute_provider", _executar)
    monkeypatch.setattr(
        regeneracao, "build_result", lambda ctx, ch, out, u, b: {"answer": out.out, "model": out.chosen}
    )
    return feitas


@pytest.mark.asyncio
async def test_failing_exploratory_answer_is_replaced_by_the_exploitation_one(monkeypatch, chamadas):
    monkeypatch.setattr(regeneracao, "reprovaria", lambda texto, **kw: True)
    escolha = _escolha()
    nova, _, resultado = await regeneracao.talvez_regenerar(
        _ctx(), escolha, ProviderOutcome("", {}, "explorado"), "P", {"grounded": False}, {"answer": ""}, 0.9
    )
    assert chamadas == ["guloso"] and resultado["model"] == "guloso" and nova.chosen == "guloso"
    regime = escolha.decision["regime"]
    assert regime["regenerado"] and regime["modelo_explorado"] == "explorado"


@pytest.mark.asyncio
async def test_passing_answers_and_exploitation_are_left_alone(monkeypatch, chamadas):
    monkeypatch.setattr(regeneracao, "reprovaria", lambda texto, **kw: False)
    out = ProviderOutcome("ok", {}, "explorado")
    assert (await regeneracao.talvez_regenerar(_ctx(), _escolha(), out, "P", {}, {"answer": "ok"}, 0.1))[2] == {
        "answer": "ok"
    }
    monkeypatch.setattr(regeneracao, "reprovaria", lambda texto, **kw: True)
    await regeneracao.talvez_regenerar(_ctx(), _escolha(explorou=False), out, "P", {}, {"answer": ""}, 0.9)
    assert chamadas == []


@pytest.mark.asyncio
async def test_a_failed_regeneration_keeps_the_explored_answer(monkeypatch):
    async def _falha(ctx, choice, prompt):
        raise RuntimeError("provedor fora")

    monkeypatch.setattr(regeneracao, "execute_provider", _falha)
    monkeypatch.setattr(regeneracao, "reprovaria", lambda texto, **kw: True)
    escolha = _escolha()
    _, _, resultado = await regeneracao.talvez_regenerar(
        _ctx(), escolha, ProviderOutcome("x", {}, "explorado"), "P", {}, {"answer": "x"}, 0.9
    )
    assert resultado == {"answer": "x"} and escolha.decision["regime"]["regeneracao_falhou"] == "RuntimeError"


def test_the_pure_check_matches_the_delivery_paths_abstention():
    from app.services.regime.verificacao import reprovaria

    assert reprovaria(
        "", incerteza=0.2, grounded=True, retrieval_mode="full_retrieval", workload_class="x", complexidade=""
    )
    assert not reprovaria(
        "Resposta completa e fundamentada.",
        incerteza=0.1,
        grounded=True,
        retrieval_mode="full_retrieval",
        workload_class="knowledge_lookup",
        complexidade="moderate",
    )
