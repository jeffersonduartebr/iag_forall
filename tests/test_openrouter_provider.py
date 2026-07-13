# Objective: Test OpenRouter provider integration and catalog helpers.
"""Tests for OpenRouter provider, catalog, and pricing integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.utils import pricing as pr

from app import model_registry as mr
from app import openrouter_catalog as orc
from app import providers_async as pa


def _fresh_registry():
    mr.ModelRegistry._instance = None
    return mr.ModelRegistry()


def test_registry_openrouter_get_or_default():
    registry = _fresh_registry()
    cfg = registry.get_or_default("openrouter/anthropic/claude-sonnet-4")
    assert cfg.provider == mr.Provider.OPENROUTER
    assert cfg.full_name == "openrouter/anthropic/claude-sonnet-4"
    assert cfg.name == "anthropic/claude-sonnet-4"


def test_is_provider_configured_openrouter(monkeypatch):
    registry = _fresh_registry()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert registry.is_provider_configured(mr.Provider.OPENROUTER) is False
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert registry.is_provider_configured(mr.Provider.OPENROUTER) is True


def test_openrouter_catalog_normalize_and_pricing():
    orc.invalidate_openrouter_catalog_cache()
    orc._CACHE["data"] = [
        {
            "id": "anthropic/claude-sonnet-4",
            "full_name": "openrouter/anthropic/claude-sonnet-4",
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        }
    ]
    orc._CACHE["expires_at"] = orc._CACHE["expires_at"] or 9999999999.0

    pricing = orc.get_openrouter_pricing_per_1k("anthropic/claude-sonnet-4")
    assert pricing["in"] == pytest.approx(0.003)
    assert pricing["out"] == pytest.approx(0.015)

    cost = pr.get_model_cost("openrouter/anthropic/claude-sonnet-4", 1000, 1000)
    assert cost == pytest.approx(0.018)


@pytest.mark.asyncio
async def test_fetch_openrouter_models_caches(monkeypatch):
    orc.invalidate_openrouter_catalog_cache()

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {
                        "id": "openai/gpt-4o-mini",
                        "description": "fast",
                        "context_length": 128000,
                        "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
                    }
                ]
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            assert "openrouter.ai" in url
            return _Resp()

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: _Client())
    models = await orc.fetch_openrouter_models(force_refresh=True)
    assert len(models) == 1
    assert models[0]["full_name"] == "openrouter/openai/gpt-4o-mini"

    cached = await orc.fetch_openrouter_models()
    assert cached == models


def test_provider_factory_returns_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    pa.reset_provider_runtime_state()
    with patch.object(pa, "OPENROUTER_API_KEY", "sk-or-test"):
        provider = pa.ProviderFactory.get_provider("openrouter/anthropic/claude-sonnet-4")
    assert isinstance(provider, pa.OpenRouterProvider)


@pytest.mark.asyncio
async def test_openrouter_provider_generate(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    usage = MagicMock(prompt_tokens=10, completion_tokens=20)
    choice = MagicMock()
    choice.message.content = "hello from openrouter"
    # Respostas reais do SDK sempre trazem estes campos com tipos concretos.
    choice.message.tool_calls = None
    choice.finish_reason = "stop"
    resp = MagicMock(choices=[choice], usage=usage)
    resp.model_dump.return_value = {"choices": [{"message": {"content": "hello from openrouter"}}]}

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=resp)

    with patch.object(pa, "OPENROUTER_API_KEY", "sk-or-test"):
        with patch.object(pa, "AsyncOpenAI", return_value=mock_client):
            provider = pa.OpenRouterProvider()

    with patch.object(pa, "get_model_cost", return_value=0.01):
        result = await provider.generate("ping", model="anthropic/claude-sonnet-4")

    assert result.text == "hello from openrouter"
    assert result.model_used == "anthropic/claude-sonnet-4"
    mock_client.chat.completions.create.assert_awaited_once()
    call_kwargs = mock_client.chat.completions.create.await_args.kwargs
    assert call_kwargs["model"] == "anthropic/claude-sonnet-4"


@pytest.mark.asyncio
async def test_call_model_strips_openrouter_prefix(monkeypatch):
    class _Provider:
        async def generate(self, **kwargs):
            assert kwargs["model"] == "openai/gpt-4o-mini"
            return pa.LLMResponse(
                text="ok",
                latency=0.1,
                load_time=0.0,
                cost=0.0,
                prompt_tokens=1,
                completion_tokens=1,
                model_used=kwargs["model"],
                raw_payload="{}",
                reasoning=None,
            )

    monkeypatch.setattr(pa, "is_provider_temporarily_unavailable", lambda _m: False)
    monkeypatch.setattr(pa.ProviderFactory, "get_provider", lambda _m: _Provider())
    text, meta = await pa.call_model("openrouter/openai/gpt-4o-mini", "hi")
    assert text == "ok"
    assert meta["latency"] == 0.1
