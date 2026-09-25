# Objective: Shadow execution — pairing, judges, admissibility, cuts and failures (R4-R6, R9, R13).
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from app.services.sombra import cotas, executor
from sombra_fakes import JUIZES, Modelos, capturar_registro, job, ligar


@pytest.fixture
def rds(fake_redis):
    return fake_redis


async def _rodar(monkeypatch, rds, modelos, j=None, **cfg_kw):
    ligar(monkeypatch, **cfg_kw)
    gravado = capturar_registro(monkeypatch)
    linhas = await executor.executar(j or job(), chamar=modelos, rds=rds)
    return linhas, gravado


@pytest.mark.asyncio
async def test_every_candidate_gets_the_same_inputs_and_rag_is_not_retrieved_again(monkeypatch, rds):
    import app.rag_local as rag

    monkeypatch.setattr(rag, "build_retrieval_bundle", lambda *a, **k: (_ for _ in ()).throw(AssertionError("RAG de novo")))
    modelos = Modelos()
    linhas, gravado = await _rodar(monkeypatch, rds, modelos)
    chamadas = modelos.de_candidatas()
    assert {c["model"] for c in chamadas} == set(job()["candidatas"])
    campos = {(c["prompt"], c["system_prompt"], c["temperature"], c["max_tokens"], c["modality"]) for c in chamadas}
    assert campos == {("PROMPT FINAL COM CONTEXTO", "SYS", 0.4, 2048, "text")}
    rubricas = [c["prompt"] for c in modelos.chamadas if "### RUBRICA" in c["prompt"]]
    assert rubricas and all("CONTEXTO RAG DA REQUISICAO" in p for p in rubricas)  # o contexto da própria requisição
    assert gravado["linhas"] == linhas and [linha["papel"] for linha in linhas] == ["entregue", "sombra", "sombra"]


@pytest.mark.asyncio
async def test_judges_exclude_the_evaluated_company_and_the_panel_is_flagged_when_not_uniform(monkeypatch, rds):
    linhas, _ = await _rodar(monkeypatch, rds, Modelos())
    paineis = {linha["modelo"]: linha["painel"] for linha in linhas}
    assert "gemini/gemini-3.1-pro-preview" not in paineis["gemini/gemini-3.8-flash"]  # google não julga google
    assert "openrouter/anthropic/claude-opus-5.5" not in paineis["openrouter/anthropic/claude-sonnet-5"]
    assert set(paineis["openrouter/deepseek/deepseek-v4.1-flash"]) == set(JUIZES)
    assert all(linha["painel_uniforme"] is False for linha in linhas)


@pytest.mark.asyncio
async def test_rows_carry_scores_cost_latency_tokens_version_and_hash(monkeypatch, rds):
    modelos = Modelos(notas={"deepseek": 9.0, "sonnet": 5.0, "ENTREGUE": 7.0})
    linhas, _ = await _rodar(monkeypatch, rds, modelos)
    por = {linha["modelo"]: linha for linha in linhas}
    ds = por["openrouter/deepseek/deepseek-v4.1-flash"]
    assert ds["status"] == "ok" and ds["escore_agregado"] > por["openrouter/anthropic/claude-sonnet-5"]["escore_agregado"]
    assert (ds["custo_usd"], ds["tokens_entrada"], ds["tokens_saida"]) == (0.01, 100, 50)
    assert ds["versao_modelo"] == "openrouter/deepseek/deepseek-v4.1-flash-2026" and len(ds["sha256_texto"]) == 64
    entregue = por["gemini/gemini-3.8-flash"]
    assert (entregue["papel"], entregue["custo_usd"], entregue["regime_entrega"]) == ("entregue", 0.002, "aproveitamento")
    assert set(ds["escores_juizes"]) == set(ds["painel"])


@pytest.mark.asyncio
async def test_inadmissible_candidates_are_refused_before_any_call(monkeypatch, rds):
    modelos = Modelos()
    j = job(candidatas=["openai/gpt-5", "openrouter/x/y"])
    linhas, _ = await _rodar(monkeypatch, rds, modelos, j, provedores=("gemini/",))
    status = {linha["modelo"]: linha["status"] for linha in linhas if linha["papel"] == "sombra"}
    assert status == {"openai/gpt-5": "recusado_allowlist", "openrouter/x/y": "recusado_allowlist"}
    assert modelos.de_candidatas() == []


@pytest.mark.asyncio
async def test_a_required_region_refuses_unverifiable_and_other_regions(monkeypatch, rds):
    import app.providers._gemini as gem

    monkeypatch.setattr(gem, "_cliente_genai", lambda: SimpleNamespace(_api_client=SimpleNamespace(vertexai=True, location="global")))
    j = job(candidatas=["gemini/gemini-2.5-flash", "openrouter/deepseek/deepseek-v4.1-flash"])
    linhas, _ = await _rodar(monkeypatch, rds, Modelos(), j, regiao="southamerica-east1")
    status = {linha["modelo"]: linha["status"] for linha in linhas if linha["papel"] == "sombra"}
    assert status == {"gemini/gemini-2.5-flash": "recusado_regiao", "openrouter/deepseek/deepseek-v4.1-flash": "recusado_regiao"}
    monkeypatch.setattr(gem, "_cliente_genai", lambda: SimpleNamespace(_api_client=SimpleNamespace(vertexai=True, location="southamerica-east1")))
    from app.services.sombra.admissibilidade import motivo_recusa
    from sombra_fakes import cfg

    assert motivo_recusa("gemini/gemini-2.5-flash", cfg(regiao="southamerica-east1")) is None


@pytest.mark.asyncio
async def test_local_candidates_yield_to_real_requests_on_the_gpu(monkeypatch, rds):
    monkeypatch.setattr(cotas, "gpu_ocupada", lambda: True)
    linhas, _ = await _rodar(monkeypatch, rds, Modelos(), job(candidatas=["ollama/granite4.2:8b"]))
    assert linhas[-1]["status"] == "gpu_ocupada"


@pytest.mark.asyncio
async def test_timeouts_and_errors_are_recorded_and_swallowed(monkeypatch, rds):
    modelos = Modelos(falhas={"openrouter/anthropic/claude-sonnet-5": RuntimeError("boom"),
                              "openrouter/deepseek/deepseek-v4.1-flash": asyncio.TimeoutError()})
    linhas, _ = await _rodar(monkeypatch, rds, modelos)
    status = {linha["modelo"]: linha["status"] for linha in linhas}
    assert status["openrouter/anthropic/claude-sonnet-5"] == "erro"
    assert status["openrouter/deepseek/deepseek-v4.1-flash"] == "timeout"
    assert status["gemini/gemini-3.8-flash"] == "ok" and rds.get("shadow:inflight") not in (b"1", b"2")


@pytest.mark.asyncio
async def test_switch_off_and_missing_redis_still_record_the_sampled_request(monkeypatch, rds):
    linhas, _ = await _rodar(monkeypatch, rds, Modelos(), ligada=False)
    assert {(linha["executada"], linha["status"]) for linha in linhas} == {(False, "desligada")}
    ligar(monkeypatch)
    capturar_registro(monkeypatch)
    monkeypatch.setattr(executor, "_redis", lambda: None)
    sem = await executor.executar(job(), chamar=Modelos())
    assert {linha["status"] for linha in sem} == {"sem_redis"} and len(sem) == 3
