import os
import sys

import pytest
from fastapi import HTTPException

# Prioriza /app do projeto para resolver pacote "app" correto nos testes.
for _mod in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
    sys.modules.pop(_mod, None)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))


def _stabilize_settings_get(monkeypatch):
    from app import settings_dynamic

    monkeypatch.setattr(settings_dynamic.settings, "get", lambda _key, fallback=None: fallback)


@pytest.mark.asyncio
async def test_route_query_maps_provider_timeout_to_504(monkeypatch):
    _stabilize_settings_get(monkeypatch)
    from app.main import route_query
    from app.schemas import QueryRequest
    from app.providers_async import ProviderCallError

    async def _raise_timeout(**kwargs):
        raise ProviderCallError(model="openai/gpt-4o", message="timeout", category="provider_timeout")

    monkeypatch.setattr("app.main.route_and_answer", _raise_timeout)

    req = QueryRequest(query="teste", modality="text")
    with pytest.raises(HTTPException) as exc:
        await route_query(req)

    assert exc.value.status_code == 504


@pytest.mark.asyncio
async def test_route_query_maps_provider_rate_limit_to_429(monkeypatch):
    _stabilize_settings_get(monkeypatch)
    from app.main import route_query
    from app.schemas import QueryRequest
    from app.providers_async import ProviderCallError

    async def _raise_rate_limit(**kwargs):
        raise ProviderCallError(model="openai/gpt-4o", message="rate limited", category="provider_rate_limit")

    monkeypatch.setattr("app.main.route_and_answer", _raise_rate_limit)

    req = QueryRequest(query="teste", modality="text")
    with pytest.raises(HTTPException) as exc:
        await route_query(req)

    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_route_query_maps_provider_unavailable_to_502(monkeypatch):
    _stabilize_settings_get(monkeypatch)
    from app.main import route_query
    from app.schemas import QueryRequest
    from app.providers_async import ProviderCallError

    async def _raise_unavailable(**kwargs):
        raise ProviderCallError(model="openai/gpt-4o", message="down", category="provider_unavailable")

    monkeypatch.setattr("app.main.route_and_answer", _raise_unavailable)

    req = QueryRequest(query="teste", modality="text")
    with pytest.raises(HTTPException) as exc:
        await route_query(req)

    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_route_query_maps_circuit_open_to_503(monkeypatch):
    _stabilize_settings_get(monkeypatch)
    from app.main import route_query
    from app.schemas import QueryRequest
    from app.providers_async import ProviderCircuitOpenError

    async def _raise_circuit(**kwargs):
        raise ProviderCircuitOpenError(model="openai/gpt-4o", message="open")

    monkeypatch.setattr("app.main.route_and_answer", _raise_circuit)

    req = QueryRequest(query="teste", modality="text")
    with pytest.raises(HTTPException) as exc:
        await route_query(req)

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_startup_rejects_empty_admin_token(monkeypatch):
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
