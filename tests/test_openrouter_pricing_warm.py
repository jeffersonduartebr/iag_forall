# Objective: OpenRouter models are priced from the catalog loaded at startup (not zero).
"""Production finding (2026-09-24): openrouter/deepseek/deepseek-v4.1-flash cost 0.0 per query because the
pricing catalog was only fetched by the exploration flow."""

import pytest
from app.utils import pricing

from app import openrouter_catalog as oc


@pytest.fixture(autouse=True)
def _catalogo_limpo():
    oc._CACHE.update(data=[], expires_at=0.0)
    yield
    oc._CACHE.update(data=[], expires_at=0.0)


@pytest.mark.asyncio
async def test_warm_catalog_prices_openrouter_models(monkeypatch):
    async def _buscar(*, force_refresh=False):
        oc._CACHE["data"] = [{"id": "deepseek/deepseek-v4.1-flash", "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}}]
        return oc._CACHE["data"]

    monkeypatch.setattr(oc, "get_openrouter_api_key", lambda: "sk-or-test")
    monkeypatch.setattr(oc, "fetch_openrouter_models", _buscar)
    assert await oc.warm_pricing_catalog() == 1
    precos = pricing._lookup_catalog("openrouter/deepseek/deepseek-v4.1-flash")
    assert precos == pytest.approx({"in": 0.00015, "out": 0.0006})


@pytest.mark.asyncio
async def test_no_key_no_fetch(monkeypatch):
    async def _nao_deve(**kwargs):
        raise AssertionError("não deveria buscar sem chave")

    monkeypatch.setattr(oc, "get_openrouter_api_key", lambda: "")
    monkeypatch.setattr(oc, "fetch_openrouter_models", _nao_deve)
    assert await oc.warm_pricing_catalog() == 0
