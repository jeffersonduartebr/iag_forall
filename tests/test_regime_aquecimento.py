# Objective: The warm-up before the field — higher ε over the configured candidates only, no participant cap, no
# window written — and the switch to the protocol regime on the day after REGIME_AQUECIMENTO_ATE.
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest
from app.services.regime import config, sorteio

ATE = dt.date(2026, 11, 15)


def _ctx(user_key=None):
    return SimpleNamespace(
        tenant_id="ifrn-caso1",
        query="q",
        modality="text",
        hints={"user_key": user_key} if user_key else {},
        scored_candidates=[],
        strategy_weights={},
        deps={},
    )


@pytest.fixture
def aquecendo(monkeypatch, fake_redis):
    cfg = config.ConfigRegime(("ifrn-caso1",), 0.15, 0.15, 20, aquecimento_ate=ATE, epsilon_aquecimento=0.5)
    monkeypatch.setattr(config, "carregar", lambda: cfg)
    monkeypatch.setattr(config.ConfigRegime, "em_aquecimento", lambda self, hoje=None: self.aquecimento_ate is not None)

    async def _medias(ctx):
        return {"a": 0.9, "b": 0.4, "c": 0.4}

    async def _catalogo(ctx, modelos):  # pragma: no cover - o aquecimento não pode consultar o catálogo
        raise AssertionError("catálogo consultado no aquecimento")

    async def _top2(ctx, models, u):
        return ["b", "a"], [], {}

    monkeypatch.setattr(sorteio, "_scored_top2", _top2)
    monkeypatch.setattr(sorteio, "_medias", _medias)
    monkeypatch.setattr(sorteio, "_pool_catalogo", _catalogo)
    monkeypatch.setattr(sorteio, "_redis", lambda: fake_redis)
    monkeypatch.setattr(sorteio, "_decision_record", lambda *a, **k: {})
    return fake_redis


async def _escolher(monkeypatch, rid, ctx):
    monkeypatch.setattr("app.correlation.get_correlation_id", lambda: rid)
    return await sorteio.rota_do_regime(ctx, ["a", "b", "c"], 0.3)


@pytest.mark.asyncio
async def test_warmup_explores_only_the_candidates_at_its_own_epsilon_without_a_participant(monkeypatch, aquecendo):
    escolhas = [await _escolher(monkeypatch, f"w{i}", _ctx()) for i in range(400)]
    infos = [e.decision["regime"] for e in escolhas]
    assert {i["fase"] for i in infos} == {"aquecimento"}
    assert {i["epsilon_efetivo"] for i in infos} == {0.5}
    exploradas = [e.chosen for e, i in zip(escolhas, infos) if i["explorou"]]
    assert 0.4 < len(exploradas) / 400 < 0.6
    assert set(exploradas) == {"b", "c"}  # só candidatas não gulosas, nunca o catálogo
    for e, i in zip(escolhas, infos):  # probabilidade exata: 1−ε no guloso, ε/2 em cada braço
        assert i["p_atribuicao"] == pytest.approx(0.25 if i["explorou"] else 0.5)


@pytest.mark.asyncio
async def test_warmup_leaves_the_participant_window_clean(monkeypatch, aquecendo):
    for i in range(30):
        await _escolher(monkeypatch, f"p{i}", _ctx("P1"))
    assert not list(aquecendo.scan_iter("regime:*"))


@pytest.mark.asyncio
async def test_frozen_policy_still_stops_exploration_in_warmup(monkeypatch, aquecendo):
    aquecendo.set("eval:frozen:active", "run-1")
    info = (await _escolher(monkeypatch, "f", _ctx())).decision["regime"]
    assert (info["explorou"], info["motivo_sem_exploracao"]) == (False, "politica_congelada")


def test_warmup_ends_after_the_configured_local_day():
    cfg = config.ConfigRegime((), 0.15, 0.15, 20, aquecimento_ate=ATE)
    assert cfg.em_aquecimento(ATE) and not cfg.em_aquecimento(ATE + dt.timedelta(days=1))
    assert not config.ConfigRegime((), 0.15, 0.15, 20).em_aquecimento()
    assert cfg.em_aquecimento() is (dt.datetime.now(config.ZoneInfo(config.FUSO)).date() <= ATE)


@pytest.mark.parametrize("valor, esperado", [("2026-11-15", ATE), ("", None), ("amanhã", None)])
def test_warmup_date_is_read_from_settings(monkeypatch, valor, esperado):
    valores = {"REGIME_AQUECIMENTO_ATE": valor, "REGIME_EPSILON_AQUECIMENTO": "7"}
    monkeypatch.setattr(config.settings, "get", lambda k, d=None: valores.get(k, d))
    cfg = config.carregar()
    assert (cfg.aquecimento_ate, cfg.epsilon_aquecimento) == (esperado, 1.0)
