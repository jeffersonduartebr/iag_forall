"""Tests for provider error mapping in query_runtime and main wiring."""

import os
import sys

import pytest
from fastapi import HTTPException

for _mod in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
    sys.modules.pop(_mod, None)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))


def _stabilize_settings_get(monkeypatch):
    """Keep dynamic settings deterministic during HTTP/service tests."""
    from app import settings_dynamic

    monkeypatch.setattr(settings_dynamic.settings, "get", lambda _key, fallback=None: fallback)


def _make_request():
    from app.schemas import QueryRequest

    return QueryRequest(query="teste", modality="text")


@pytest.mark.asyncio
async def test_process_query_request_maps_provider_timeout_to_504(monkeypatch):
    """Provider timeout must map to HTTP 504 in query runtime."""
    _stabilize_settings_get(monkeypatch)
    from app.providers_async import ProviderCallError
    from app.services import query_runtime as qr

    async def _raise_timeout(**kwargs):
        raise ProviderCallError(model="openai/gpt-4o", message="timeout", category="provider_timeout")

    monkeypatch.setattr(qr, "route_and_answer", _raise_timeout)
    monkeypatch.setattr(qr, "check_input_guardrails", lambda _q: type("D", (), {"allowed": True, "reasons": []})())
    monkeypatch.setattr(qr, "check_tenant_budget", lambda _tenant: type("B", (), {"allowed": True})())

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
    monkeypatch.setattr(qr, "check_input_guardrails", lambda _q: type("D", (), {"allowed": True, "reasons": []})())
    monkeypatch.setattr(qr, "check_tenant_budget", lambda _tenant: type("B", (), {"allowed": True})())

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
    monkeypatch.setattr(qr, "check_input_guardrails", lambda _q: type("D", (), {"allowed": True, "reasons": []})())
    monkeypatch.setattr(qr, "check_tenant_budget", lambda _tenant: type("B", (), {"allowed": True})())

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
    monkeypatch.setattr(qr, "check_input_guardrails", lambda _q: type("D", (), {"allowed": True, "reasons": []})())
    monkeypatch.setattr(qr, "check_tenant_budget", lambda _tenant: type("B", (), {"allowed": True})())

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
