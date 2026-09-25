# Objective: No shadow answer text leaves memory: not in rows, logs, Redis, metrics or spans (R7).
from __future__ import annotations

import json
import logging

import pytest
from app.services.sombra import executor
from sombra_fakes import Modelos, capturar_registro, job, ligar

SENTINELA = "SENTINELA-TEXTO-QUE-NAO-PODE-VAZAR-7f3a"


@pytest.mark.asyncio
async def test_shadow_answer_text_is_discarded_after_scoring(monkeypatch, fake_redis, caplog):
    from app.observability import registry
    from prometheus_client import generate_latest

    ligar(monkeypatch)
    gravado = capturar_registro(monkeypatch)
    caplog.set_level(logging.DEBUG)
    modelos = Modelos(resposta=SENTINELA, falhas={"openrouter/anthropic/claude-sonnet-5": RuntimeError(SENTINELA)})
    linhas = await executor.executar(job(), chamar=modelos, rds=fake_redis)

    assert any(SENTINELA in c["prompt"] for c in modelos.chamadas)  # os juízes leram o texto...
    assert SENTINELA not in json.dumps(gravado["linhas"], default=str)  # ...mas ele não foi gravado
    assert all(linha.get("sha256_texto") or linha["status"] != "ok" for linha in linhas)
    assert SENTINELA not in caplog.text  # nem pela mensagem de erro do provedor
    for chave in fake_redis.keys():
        assert SENTINELA.encode() not in (fake_redis.dump(chave) or b"")
    assert SENTINELA not in generate_latest(registry).decode()


@pytest.mark.asyncio
async def test_shadow_calls_open_no_span_with_the_text(monkeypatch, fake_redis):
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exportador = InMemorySpanExporter()
    provedor = TracerProvider()
    provedor.add_span_processor(SimpleSpanProcessor(exportador))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provedor, raising=False)
    ligar(monkeypatch)
    capturar_registro(monkeypatch)
    await executor.executar(job(), chamar=Modelos(resposta=SENTINELA), rds=fake_redis)
    for span in exportador.get_finished_spans():
        assert SENTINELA not in json.dumps(dict(span.attributes or {}), default=str)
