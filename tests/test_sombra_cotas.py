# Objective: Shadow budget, per-tenant hourly cap, local-day rollover, suspensions and separate cost (R9-R11).
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest
from app.services.sombra import cotas, executor
from app.services.sombra.metricas import SHADOW_BUDGET_EXHAUSTED_HOUR, SHADOW_COST
from sombra_fakes import Modelos, capturar_registro, cfg, job, ligar

FORTALEZA = ZoneInfo("America/Fortaleza")


def _relogio(monkeypatch, momento):
    atual = {"t": momento}
    monkeypatch.setattr(cotas, "agora", lambda c: atual["t"])
    return atual


def test_the_budget_day_turns_at_local_midnight_not_utc(monkeypatch, fake_redis):
    c = cfg(orcamento_usd=1.0)
    relogio = _relogio(monkeypatch, dt.datetime(2026, 11, 20, 23, 30, tzinfo=FORTALEZA))  # 02:30 UTC do dia 21
    cotas.somar_gasto(fake_redis, c, 1.0)
    assert cotas.orcamento_esgotado(fake_redis, c)
    relogio["t"] = dt.datetime(2026, 11, 21, 0, 5, tzinfo=FORTALEZA)
    assert not cotas.orcamento_esgotado(fake_redis, c)  # retomada automática no dia seguinte (fuso local)
    assert SHADOW_BUDGET_EXHAUSTED_HOUR._value.get() == -1


def test_exhaustion_hour_is_published_only_by_the_call_that_crossed_the_budget(monkeypatch, fake_redis):
    c = cfg(orcamento_usd=0.05)
    _relogio(monkeypatch, dt.datetime(2026, 11, 20, 14, 30, tzinfo=FORTALEZA))
    assert cotas.somar_gasto(fake_redis, c, 0.03) is None
    assert cotas.somar_gasto(fake_redis, c, 0.03) == pytest.approx(14.5)
    assert cotas.somar_gasto(fake_redis, c, 0.03) is None


def test_the_per_tenant_cap_is_hourly_and_per_tenant(monkeypatch, fake_redis):
    c = cfg(teto_tenant_hora=2)
    relogio = _relogio(monkeypatch, dt.datetime(2026, 11, 20, 9, 10, tzinfo=FORTALEZA))
    assert [cotas.contar_amostra_tenant(fake_redis, c, "a") for _ in range(3)] == [True, True, False]
    assert cotas.contar_amostra_tenant(fake_redis, c, "b")
    relogio["t"] = dt.datetime(2026, 11, 20, 10, 0, tzinfo=FORTALEZA)
    assert cotas.contar_amostra_tenant(fake_redis, c, "a")


def test_global_concurrency_slots(fake_redis):
    c = cfg(max_concorrencia=2)
    assert cotas.tomar_vaga(fake_redis, c) and cotas.tomar_vaga(fake_redis, c) and not cotas.tomar_vaga(fake_redis, c)
    cotas.soltar_vaga(fake_redis)
    assert cotas.tomar_vaga(fake_redis, c)


@pytest.mark.asyncio
async def test_cut_requests_are_recorded_with_one_row_per_configuration_and_a_suspension(monkeypatch, fake_redis):
    ligar(monkeypatch, orcamento_usd=0.0)
    gravado = capturar_registro(monkeypatch)
    relogio = _relogio(monkeypatch, dt.datetime(2026, 11, 20, 11, 0, tzinfo=FORTALEZA))
    linhas = await executor.executar(job(), chamar=Modelos(), rds=fake_redis)
    assert len(linhas) == 3 and {(linha["executada"], linha["status"], linha["motivo_corte"]) for linha in linhas} == {
        (False, "orcamento_esgotado", "orcamento_esgotado")
    }
    assert gravado["abertas"] == ["orcamento_esgotado:*:20261120"]
    ligar(monkeypatch)  # orçamento de volta no dia seguinte: a suspensão vencida é fechada
    relogio["t"] = dt.datetime(2026, 11, 21, 8, 0, tzinfo=FORTALEZA)
    await executor.executar(job(), chamar=Modelos(), rds=fake_redis)
    assert gravado["fechadas"] == ["orcamento_esgotado:*:20261120"]


@pytest.mark.asyncio
async def test_tenant_cap_cuts_are_recorded(monkeypatch, fake_redis):
    ligar(monkeypatch, teto_tenant_hora=0)
    capturar_registro(monkeypatch)
    linhas = await executor.executar(job(), chamar=Modelos(), rds=fake_redis)
    assert {linha["status"] for linha in linhas} == {"teto_tenant"}


@pytest.mark.asyncio
async def test_shadow_cost_goes_to_its_own_counter_and_budget(monkeypatch, fake_redis):
    from app.providers_async import PROV_COST

    ligar(monkeypatch)
    capturar_registro(monkeypatch)
    antes_sistema = sum(s.value for m in PROV_COST.collect() for s in m.samples)
    antes = SHADOW_COST.labels(model="openrouter/deepseek/deepseek-v4.1-flash")._value.get()
    await executor.executar(job(), chamar=Modelos(), rds=fake_redis)
    assert SHADOW_COST.labels(model="openrouter/deepseek/deepseek-v4.1-flash")._value.get() > antes
    assert sum(s.value for m in PROV_COST.collect() for s in m.samples) == antes_sistema
    assert cotas.gasto_hoje(fake_redis, cfg()) > 0
