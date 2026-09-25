# Objective: The regime's explicit draw — exploitation is greedy by posterior mean, exploration is uniform over
# admissible arms, and the assignment probability recorded is exact.
from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.services.regime import config, sorteio
from app.services.route_decision import Candidate


def _ctx(user_key="P1", episode_id="E1", tenant="ifrn-caso1"):
    return SimpleNamespace(
        tenant_id=tenant,
        query="q",
        modality="text",
        hints={"user_key": user_key, "episode_id": episode_id},
        scored_candidates=[],
        strategy_weights={},
        deps={},
    )


@pytest.fixture
def regime(monkeypatch, fake_redis):
    monkeypatch.setattr(config, "carregar", lambda: config.ConfigRegime(("ifrn-caso1",), 0.15, 0.15, 20))
    pontuados = [
        Candidate(model=m, quality=1, latency_s=1, cost_usd=0, risk=0, score=s)
        for m, s in (("b", 0.9), ("a", 0.8), ("c", 0.7))
    ]

    async def _top2(ctx, models, u):
        return ["b", "a"], pontuados, {"w_quality": 1.0}

    async def _medias(ctx):
        return {"a": 0.9, "b": 0.4, "c": 0.4}

    async def _catalogo(ctx, modelos):
        return ["openrouter/x/cat1"]

    monkeypatch.setattr(sorteio, "_scored_top2", _top2)
    monkeypatch.setattr(sorteio, "_medias", _medias)
    monkeypatch.setattr(sorteio, "_pool_catalogo", _catalogo)
    monkeypatch.setattr(sorteio, "_redis", lambda: fake_redis)
    monkeypatch.setattr(sorteio, "_decision_record", lambda *a, **k: {})
    return fake_redis


async def _escolher(monkeypatch, rid, ctx=None):
    monkeypatch.setattr("app.correlation.get_correlation_id", lambda: rid)
    return await sorteio.rota_do_regime(ctx or _ctx(), ["a", "b", "c"], 0.3)


@pytest.mark.asyncio
async def test_other_tenants_keep_normal_routing(monkeypatch, regime):
    assert await _escolher(monkeypatch, "r", _ctx(tenant="outro")) is None


@pytest.mark.asyncio
async def test_new_participants_are_exploited_until_the_window_allows_exploring(monkeypatch, regime):
    escolha = await _escolher(monkeypatch, "r0")
    info = escolha.decision["regime"]
    assert escolha.chosen == "a" and info["aproveitamento"] == "a"  # maior média posterior
    assert (info["explorou"], info["epsilon_efetivo"], info["p_atribuicao"], info["motivo_sem_exploracao"]) == (
        False,
        0.0,
        1.0,
        "teto_janela",
    )


@pytest.mark.asyncio
async def test_probabilities_are_exact_and_the_exploration_rate_matches_epsilon(monkeypatch, regime):
    for i in range(6):  # janela com 6 aproveitamentos: explorar passa a caber no teto
        sorteio.janela.registrar(regime, "P1", f"old{i}", 20, explorou=False)
    exploradas, total = [], 3000
    for i in range(total):
        rid = f"req-{i}"
        u = sorteio.uniforme(rid, "explorar")
        info = {"explorou": u < 0.15}
        if info["explorou"]:
            exploradas.append(rid)
    assert abs(len(exploradas) / total - 0.15) < 0.02  # o sorteio determinístico tem a taxa nominal
    escolha = await _escolher(monkeypatch, exploradas[0])
    info = escolha.decision["regime"]
    assert info["explorou"] and escolha.chosen in {"b", "c", "openrouter/x/cat1"} and info["k_bracos"] == 3
    assert info["p_atribuicao"] == pytest.approx(0.15 / 3) and info["epsilon_efetivo"] == 0.15
    assert escolha.exploration_mode == (escolha.chosen == "openrouter/x/cat1")


@pytest.mark.asyncio
async def test_the_draw_is_reproducible_from_the_request_id(monkeypatch, regime):
    assert sorteio.uniforme("abc", "braco") == sorteio.uniforme("abc", "braco") != sorteio.uniforme("abd", "braco")


@pytest.mark.asyncio
@pytest.mark.parametrize(("ctx", "motivo"), [(_ctx(user_key=None), "sem_participante")])
async def test_no_participant_means_no_exploration(monkeypatch, regime, ctx, motivo):
    escolha = await _escolher(monkeypatch, "r", ctx)
    assert escolha.decision["regime"]["motivo_sem_exploracao"] == motivo and escolha.chosen == "a"


@pytest.mark.asyncio
async def test_a_frozen_policy_disables_exploration(monkeypatch, regime):
    regime.set("eval:frozen:active", "run-9")
    for i in range(10):
        sorteio.janela.registrar(regime, "P1", f"old{i}", 20, explorou=False)
    escolha = await _escolher(monkeypatch, "r")
    assert escolha.decision["regime"]["motivo_sem_exploracao"] == "politica_congelada"
