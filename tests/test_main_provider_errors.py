# Objective: Test coverage for main provider errors behavior and regressions.
"""Tests for provider error mapping in query_runtime and main wiring."""

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))


@pytest.fixture(autouse=True)
def _isolate_app_modules():
    """Reimporta os módulos ``app.*`` isoladamente por teste e restaura o estado
    global depois.

    Antes, um ``sys.modules.pop`` no nível do módulo (executado na COLEÇÃO) apagava
    a identidade de todos os módulos ``app.*`` para a suíte inteira, quebrando
    patches por string em outros arquivos. Escopar a limpeza a este arquivo, com
    restauração, mantém o isolamento sem contaminar os demais testes.
    """
    saved = {m: sys.modules[m] for m in list(sys.modules) if m == "app" or m.startswith("app.")}
    for _m in saved:
        sys.modules.pop(_m, None)
    try:
        yield
    finally:
        for _m in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
            sys.modules.pop(_m, None)
        sys.modules.update(saved)


def _stabilize_settings_get(monkeypatch):
    """Keep dynamic settings deterministic during HTTP/service tests."""
    from app import settings_dynamic

    monkeypatch.setattr(settings_dynamic.settings, "get", lambda _key, fallback=None: fallback)


async def _async_none():
    return None


async def _async_budget_allowed(_tenant):
    return type("B", (), {"allowed": True})()


async def _async_guardrails_allowed(_query):
    return type("D", (), {"allowed": True, "reasons": []})()


def _make_request():
    from app.schemas import QueryRequest

    return QueryRequest(query="teste", modality="text")


def _make_http_request(path="/query", *, defer=False, reason="ollama_overloaded", pressure_state="congested"):
    state = SimpleNamespace()
    if defer:
        state.defer_to_query_job = True
        state.query_job_reason = reason
        state.query_job_pressure_state = pressure_state
    return SimpleNamespace(state=state, url=SimpleNamespace(path=path))


@pytest.mark.asyncio
async def test_process_query_request_maps_provider_timeout_to_504(monkeypatch):
    """Provider timeout must map to HTTP 504 in query runtime."""
    _stabilize_settings_get(monkeypatch)
    from app.providers_async import ProviderCallError
    from app.services import query_runtime as qr

    async def _raise_timeout(**kwargs):
        raise ProviderCallError(model="openai/gpt-4o", message="timeout", category="provider_timeout")

    monkeypatch.setattr(qr, "route_and_answer", _raise_timeout)
    monkeypatch.setattr(qr, "check_input_guardrails_async", _async_guardrails_allowed)
    monkeypatch.setattr(qr, "check_tenant_budget", _async_budget_allowed)
    monkeypatch.setattr(qr, "get_active_policy", _async_none)

    with pytest.raises(HTTPException) as exc:
        await qr.process_query_request(_make_request())
    assert exc.value.status_code == 504


@pytest.mark.asyncio
async def test_process_query_request_maps_provider_rate_limit_to_429(monkeypatch):
    """Provider rate limit must map to HTTP 429 in query runtime."""
    _stabilize_settings_get(monkeypatch)
    from app.providers_async import ProviderCallError
    from app.services import query_runtime as qr

    async def _raise_rate_limit(**kwargs):
        raise ProviderCallError(model="openai/gpt-4o", message="rate limited", category="provider_rate_limit")

    monkeypatch.setattr(qr, "route_and_answer", _raise_rate_limit)
    monkeypatch.setattr(qr, "check_input_guardrails_async", _async_guardrails_allowed)
    monkeypatch.setattr(qr, "check_tenant_budget", _async_budget_allowed)
    monkeypatch.setattr(qr, "get_active_policy", _async_none)

    with pytest.raises(HTTPException) as exc:
        await qr.process_query_request(_make_request())
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_process_query_request_maps_provider_unavailable_to_502(monkeypatch):
    """Provider unavailability must map to HTTP 502 in query runtime."""
    _stabilize_settings_get(monkeypatch)
    from app.providers_async import ProviderCallError
    from app.services import query_runtime as qr

    async def _raise_unavailable(**kwargs):
        raise ProviderCallError(model="openai/gpt-4o", message="down", category="provider_unavailable")

    monkeypatch.setattr(qr, "route_and_answer", _raise_unavailable)
    monkeypatch.setattr(qr, "check_input_guardrails_async", _async_guardrails_allowed)
    monkeypatch.setattr(qr, "check_tenant_budget", _async_budget_allowed)
    monkeypatch.setattr(qr, "get_active_policy", _async_none)

    with pytest.raises(HTTPException) as exc:
        await qr.process_query_request(_make_request())
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_process_query_request_maps_circuit_open_to_503(monkeypatch):
    """Circuit-open provider error must map to HTTP 503 in query runtime."""
    _stabilize_settings_get(monkeypatch)
    from app.providers_async import ProviderCircuitOpenError
    from app.services import query_runtime as qr

    async def _raise_circuit(**kwargs):
        raise ProviderCircuitOpenError(model="openai/gpt-4o", message="open")

    monkeypatch.setattr(qr, "route_and_answer", _raise_circuit)
    monkeypatch.setattr(qr, "check_input_guardrails_async", _async_guardrails_allowed)
    monkeypatch.setattr(qr, "check_tenant_budget", _async_budget_allowed)
    monkeypatch.setattr(qr, "get_active_policy", _async_none)

    with pytest.raises(HTTPException) as exc:
        await qr.process_query_request(_make_request())
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_route_query_uses_main_wiring(monkeypatch):
    """main.route_query should delegate to process_query_request imported in main."""
    _stabilize_settings_get(monkeypatch)
    from app import main

    async def _process(_req):
        return {
            "result": {
                "answer": "ok",
                "model": "ollama/test",
                "modality": "text",
                "latency_s": 0.1,
                "cost_per_1k": 0.0,
                "route": {"chosen_model": "ollama/test", "objectives": {}, "pareto_front": [], "explanation": "", "fallback": {"used": False, "models_tried": [], "errors": []}},
                "candidates": [],
                "metadata": {"raw_payload": None, "prompt_tokens": 0, "completion_tokens": 0},
            },
            "image_input": None,
            "selected_policy": None,
            "assigned_variant": None,
            "modality": "text",
        }

    monkeypatch.setattr(main, "process_query_request", _process)
    monkeypatch.setattr(main, "record_query_side_effects", lambda req, result, image_input: None)
    out = await main.route_query(_make_request())
    assert out.answer == "ok"


@pytest.mark.asyncio
async def test_route_query_enqueues_async_job_when_request_marked(monkeypatch):
    """main.route_query should return HTTP 202 with the accepted job payload when middleware defers execution."""
    _stabilize_settings_get(monkeypatch)
    from app.schemas import QueuedQueryAcceptedResponse

    from app import main

    monkeypatch.setattr(
        main,
        "enqueue_query_job",
        lambda **kwargs: QueuedQueryAcceptedResponse(
            job_id="job-1",
            poll_url="/query/jobs/job-1",
            result_url="/query/jobs/job-1/result",
            expires_at=123.0,
        ),
    )

    out = await main.route_query(_make_request(), _make_http_request(defer=True))
    assert out.status_code == 202
    assert b"job-1" in out.body


@pytest.mark.asyncio
async def test_route_query_proactively_enqueues_simple_query_under_pressure(monkeypatch):
    """High-priority simple queries should be deferred early when Ollama is already near capacity."""
    _stabilize_settings_get(monkeypatch)
    from app.schemas import QueuedQueryAcceptedResponse

    from app import main

    monkeypatch.setattr(
        main,
        "get_ollama_admission_snapshot",
        lambda: {
            "current_limit": 5,
            "total_inflight": 4,
            "max_queue_wait_ms": 180.0,
            "utilization": 0.8,
            "vram_ratio": 0.0,
            "pressure_state": "elevated",
        },
    )
    monkeypatch.setattr(main.settings, "get", lambda key, fallback=None: {"ADAPTIVE_LIMITER_SYNC_QUEUE_WAIT_MS": "250"}.get(key, fallback))
    monkeypatch.setattr(
        main,
        "enqueue_query_job",
        lambda **kwargs: QueuedQueryAcceptedResponse(
            job_id="job-2",
            poll_url="/query/jobs/job-2",
            result_url="/query/jobs/job-2/result",
            expires_at=123.0,
        ),
    )

    async def _should_not_run(_req):
        raise AssertionError("process_query_request should not run when proactive defer triggers")

    monkeypatch.setattr(main, "process_query_request", _should_not_run)

    request = _make_http_request()
    out = await main.route_query(
        _make_request().model_copy(update={"query": "Explique em uma frase o que e a energia solar."}),
        request,
    )
    assert out.status_code == 202
    assert b"job-2" in out.body


@pytest.mark.asyncio
async def test_route_query_proactively_enqueues_simple_query_with_two_slot_reserve(monkeypatch):
    """Short interactive workloads should defer earlier to preserve two sync slots under pressure."""
    _stabilize_settings_get(monkeypatch)
    from app.schemas import QueuedQueryAcceptedResponse

    from app import main

    monkeypatch.setattr(
        main,
        "get_ollama_admission_snapshot",
        lambda: {
            "current_limit": 5,
            "total_inflight": 3,
            "max_queue_wait_ms": 110.0,
            "utilization": 0.7,
            "vram_ratio": 0.0,
            "pressure_state": "elevated",
        },
    )
    monkeypatch.setattr(main.settings, "get", lambda key, fallback=None: {"ADAPTIVE_LIMITER_SYNC_QUEUE_WAIT_MS": "250"}.get(key, fallback))
    monkeypatch.setattr(
        main,
        "enqueue_query_job",
        lambda **kwargs: QueuedQueryAcceptedResponse(
            job_id="job-3",
            poll_url="/query/jobs/job-3",
            result_url="/query/jobs/job-3/result",
            expires_at=123.0,
        ),
    )

    async def _should_not_run(_req):
        raise AssertionError("process_query_request should not run when the two-slot reserve triggers")

    monkeypatch.setattr(main, "process_query_request", _should_not_run)

    request = _make_http_request()
    out = await main.route_query(
        _make_request().model_copy(update={"query": "Explique em uma frase o que e a agua potavel."}),
        request,
    )
    assert out.status_code == 202
    assert b"job-3" in out.body


@pytest.mark.asyncio
async def test_startup_rejects_empty_admin_token(monkeypatch):
    """startup_event should still reject an empty admin token."""
    _stabilize_settings_get(monkeypatch)
    from app import main

    def _fake_get(key, fallback=None):
        if key == "ADMIN_TOKEN":
            return ""
        return fallback

    monkeypatch.setattr(main.settings, "get", _fake_get)

    with pytest.raises(RuntimeError) as exc:
        await main.startup_event()
    assert "ADMIN_TOKEN" in str(exc.value)
