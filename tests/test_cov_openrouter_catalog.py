# Objective: OpenRouter catalog fetch/cache, key resolution, tool support and per-1k pricing.
"""Behaviour of ``app.openrouter_catalog`` with httpx mocked by ``MockTransport`` (no network)."""

from __future__ import annotations

import httpx
import pytest

from app import openrouter_catalog as cat


@pytest.fixture(autouse=True)
def _clean_cache():
    cat.invalidate_openrouter_catalog_cache()
    yield
    cat.invalidate_openrouter_catalog_cache()


@pytest.fixture
def no_dynamic_key(monkeypatch):
    from app.settings_dynamic import settings

    monkeypatch.setattr(settings, "get", lambda key, default=None: "" if key == "OPENROUTER_API_KEY" else default)


@pytest.fixture
def api(monkeypatch):
    """Serve ``api.payload`` (or ``api.status``) for every catalog request; records request headers."""
    state = type("Api", (), {"payload": {"data": []}, "status": 200, "calls": 0, "headers": None})()
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        state.calls += 1
        state.headers = request.headers
        return httpx.Response(state.status, json=state.payload)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    return state


CATALOG = {
    "data": [
        {"id": "acme/tool-model", "pricing": {"prompt": "0.000001", "completion": "0.000002"},
         "supported_parameters": ["tools", "temperature"], "context_length": 8192},
        {"id": "acme/chat-only", "pricing": {"prompt": "bad"}},
        {"id": "", "pricing": {}},
        "not-a-dict",
    ]
}


def test_api_key_prefers_dynamic_settings_over_env(monkeypatch):
    from app.settings_dynamic import settings

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    monkeypatch.setattr(settings, "get", lambda key, default=None: " sk-dyn " if key == "OPENROUTER_API_KEY" else default)
    assert cat.get_openrouter_api_key() == "sk-dyn"


def test_api_key_falls_back_to_env_when_settings_fail(monkeypatch):
    from app.settings_dynamic import settings

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(settings, "get", boom)
    monkeypatch.setenv("OPENROUTER_API_KEY", " sk-env ")
    assert cat.get_openrouter_api_key() == "sk-env"


def test_mask_hides_the_middle_of_the_key():
    assert cat.mask_openrouter_api_key("sk-or-v1-abcdef123456") == "sk-or-v1...3456"
    assert cat.mask_openrouter_api_key("short") == ""
    assert cat.mask_openrouter_api_key(None) == ""


@pytest.mark.asyncio
async def test_fetch_normalizes_sends_headers_and_caches(api, monkeypatch, no_dynamic_key):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123456")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://aristo.example")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "ARISTO")
    api.payload = CATALOG

    models = await cat.fetch_openrouter_models()
    assert [m["full_name"] for m in models] == ["openrouter/acme/tool-model", "openrouter/acme/chat-only"]
    assert models[1]["supported_parameters"] == []
    assert api.headers["authorization"] == "Bearer sk-test-123456"
    assert api.headers["http-referer"] == "https://aristo.example" and api.headers["x-title"] == "ARISTO"

    await cat.fetch_openrouter_models()
    assert api.calls == 1  # segunda chamada servida do cache
    await cat.fetch_openrouter_models(force_refresh=True)
    assert api.calls == 2


@pytest.mark.asyncio
async def test_fetch_without_key_or_optional_headers(api, monkeypatch, no_dynamic_key):
    for var in ("OPENROUTER_API_KEY", "OPENROUTER_HTTP_REFERER", "OPENROUTER_APP_NAME"):
        monkeypatch.delenv(var, raising=False)
    api.payload = {"data": "not-a-list"}
    assert await cat.fetch_openrouter_models() == []
    assert "authorization" not in api.headers and "x-title" not in api.headers


@pytest.mark.asyncio
async def test_fetch_error_serves_stale_cache_then_empty(api):
    api.payload = CATALOG
    assert len(await cat.fetch_openrouter_models()) == 2
    api.status = 503
    stale = await cat.fetch_openrouter_models(force_refresh=True)
    assert [m["id"] for m in stale] == ["acme/tool-model", "acme/chat-only"]
    cat.invalidate_openrouter_catalog_cache()
    assert await cat.fetch_openrouter_models() == []


@pytest.mark.asyncio
async def test_fetch_tolerates_non_dict_payload(api):
    api.payload = ["x"]
    assert await cat.fetch_openrouter_models() == []


@pytest.mark.asyncio
async def test_tool_support_and_pricing_read_the_cached_catalog(api):
    assert cat.openrouter_supports_tools("acme/tool-model") is False  # cache vazio: conservador
    api.payload = CATALOG
    await cat.fetch_openrouter_models()
    assert cat.openrouter_supports_tools("acme/tool-model") is True
    assert cat.openrouter_supports_tools("acme/chat-only") is False
    assert cat.openrouter_supports_tools("") is False
    assert cat.get_openrouter_pricing_per_1k("acme/tool-model") == pytest.approx({"in": 0.001, "out": 0.002})
    assert cat.get_openrouter_pricing_per_1k("acme/chat-only") is None  # preço ilegível
    assert cat.get_openrouter_pricing_per_1k("acme/missing") is None
    assert cat.get_openrouter_pricing_per_1k(" ") is None


def test_supports_tools_rejects_non_iterable_parameters():
    cat._CACHE["data"] = [{"id": "x/y", "supported_parameters": 5}]
    assert cat.openrouter_supports_tools("x/y") is False


def test_model_ids_are_fully_qualified():
    rows = [{"id": "a/b", "full_name": "openrouter/a/b"}, {"id": "c/d"}, {"id": ""}]
    assert cat.openrouter_model_ids(rows) == ["openrouter/a/b", "openrouter/c/d"]
    assert cat.openrouter_model_ids() == []


@pytest.mark.asyncio
async def test_warm_pricing_catalog_only_when_configured(api, monkeypatch, no_dynamic_key):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert await cat.warm_pricing_catalog() == 0
    assert api.calls == 0
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123456")
    api.payload = CATALOG
    assert await cat.warm_pricing_catalog() == 2
