# Objective: Test coverage for providers async extra behavior and regressions.
"""Test coverage for providers async extra behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


import asyncio
from types import SimpleNamespace

import httpx
import pybreaker
import pytest

from app import providers_async as pa


def test_timeout_quality_tokens_and_factory(monkeypatch):
    """Testa timeout quality tokens and factory."""
    monkeypatch.setattr(
        pa,
        "_runtime_provider_settings",
        lambda: {
            "adaptive_timeout_enabled": True,
            "base_timeout": 10,
            "max_timeout": 1000,
            "timeout_multiplier": 2.0,
            "reasoning_multiplier": 3.0,
        },
    )

    assert pa._get_adaptive_timeout("deepseek-r1") == 30
    assert pa._get_adaptive_timeout("llama-70b") == 20
    assert pa._get_adaptive_timeout("model-x") == 15

    monkeypatch.setattr(
        pa,
        "_runtime_provider_settings",
        lambda: {
            "adaptive_timeout_enabled": False,
            "base_timeout": 10,
            "max_timeout": 1000,
            "timeout_multiplier": 2.0,
            "reasoning_multiplier": 3.0,
        },
    )
    assert pa._get_adaptive_timeout("any") == 1000

    assert pa.heuristic_quality_estimate("") == 0.0
    assert pa.heuristic_quality_estimate("texto. " * 50) > 4.0
    assert pa._estimate_tokens("") == 0
    assert pa._estimate_tokens("abcd") >= 1

    monkeypatch.setattr(pa, "OpenAIProvider", lambda: "openai-provider")
    monkeypatch.setattr(pa, "AnthropicProvider", lambda: "anthropic-provider")
    monkeypatch.setattr(pa, "GeminiProvider", lambda: "gemini-provider")
    monkeypatch.setattr(pa, "OllamaProvider", lambda: "ollama-provider")
    monkeypatch.setattr(pa, "is_provider_configured", lambda prefix: prefix != "xpto")
    monkeypatch.setattr(pa.ProviderFactory, "_instances", {})
    assert pa.ProviderFactory.get_provider("openai/gpt-4o") == "openai-provider"
    assert pa.ProviderFactory.get_provider("ollama/phi4:latest") == "ollama-provider"

    with pytest.raises(RuntimeError):
        pa.ProviderFactory.get_provider("xpto/model")

    monkeypatch.setattr(pa, "is_provider_configured", lambda prefix: prefix == "ollama")
    with pytest.raises(RuntimeError):
        pa.ProviderFactory.get_provider("openai/gpt-4o")


@pytest.mark.asyncio
async def test_call_model_success_and_error_categories(monkeypatch):
    """Testa call model success and error categories."""
    class _P:
        """Represent `_P` within this module.

The class groups the state and behavior required for P."""
        async def generate(self, **kwargs):
            """Execute the generate routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return pa.LLMResponse(
                text="ok",
                latency=0.1,
                load_time=0.0,
                cost=0.01,
                prompt_tokens=10,
                completion_tokens=20,
                model_used="x",
                raw_payload="{}",
                reasoning=None,
            )

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", lambda m: _P())
    txt, meta = await pa.call_model("openai/gpt-4o", "hello")
    assert txt == "ok"
    assert meta["prompt_tokens"] == 10

    class _PCircuit:
        """Represent `_PCircuit` within this module.

The class groups the state and behavior required for PCircuit."""
        async def generate(self, **kwargs):
            """Execute the generate routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            raise pybreaker.CircuitBreakerError("open")

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", lambda m: _PCircuit())
    with pytest.raises(pa.ProviderCircuitOpenError):
        await pa.call_model("openai/gpt-4o", "x")

    class _PTimeout:
        """Represent `_PTimeout` within this module.

The class groups the state and behavior required for PTimeout."""
        async def generate(self, **kwargs):
            """Execute the generate routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            raise asyncio.TimeoutError("timeout")

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", lambda m: _PTimeout())
    with pytest.raises(pa.ProviderCallError) as exc1:
        await pa.call_model("openai/gpt-4o", "x")
    assert exc1.value.category == "provider_timeout"

    class RateLimitError(Exception):
        """Represent `RateLimitError` within this module.

The class groups the state and behavior required for RateLimitError."""
        pass

    class _PRL:
        """Represent `_PRL` within this module.

The class groups the state and behavior required for PRL."""
        async def generate(self, **kwargs):
            """Execute the generate routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            raise RateLimitError("rl")

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", lambda m: _PRL())
    with pytest.raises(pa.ProviderCallError) as exc2:
        await pa.call_model("openai/gpt-4o", "x")
    assert exc2.value.category == "provider_rate_limit"


@pytest.mark.asyncio
async def test_ensure_ollama_model_async_paths(monkeypatch):
    """Testa ensure ollama model async paths."""
    class _Resp:
        """Represent `_Resp` within this module.

The class groups the state and behavior required for Resp."""
        def __init__(self, code, payload=None):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self.status_code = code
            self._payload = payload or {}

        def json(self):
            """Execute the json routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return self._payload

        def raise_for_status(self):
            """Raise a synthetic HTTP error when the mocked status is unsuccessful."""
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("bad status", request=None, response=None)

    class _Client:
        """Represent `_Client` within this module.

The class groups the state and behavior required for Client."""
        def __init__(self, tags_resp, pull_resp):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self.tags_resp = tags_resp
            self.pull_resp = pull_resp

        async def get(self, *a, **k):
            """Execute the get routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return self.tags_resp

        async def post(self, *a, **k):
            """Execute the post routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return self.pull_resp

    async def _c1():
        """Execute the c1 routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return _Client(_Resp(200, {"models": [{"name": "phi4:latest"}]}), _Resp(200))

    monkeypatch.setattr(pa, "get_http_client", _c1)
    assert await pa._ensure_ollama_model_async("phi4:latest") is True

    async def _c2():
        """Execute the c2 routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return _Client(_Resp(200, {"models": []}), _Resp(200))

    monkeypatch.setattr(pa, "get_http_client", _c2)
    assert await pa._ensure_ollama_model_async("phi4:latest") is True

    async def _c3():
        """Execute the c3 routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return _Client(_Resp(500, {}), _Resp(500))

    monkeypatch.setattr(pa, "get_http_client", _c3)
    assert await pa._ensure_ollama_model_async("phi4:latest") is False

    async def _boom():
        """Execute the boom routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        raise httpx.ConnectError("x")

    monkeypatch.setattr(pa, "get_http_client", _boom)
    assert await pa._ensure_ollama_model_async("phi4:latest") is False


@pytest.mark.asyncio
async def test_fetch_ollama_tags_cache_and_probe(monkeypatch):
    """fetch_ollama_tags should reuse cached tags and expose a lightweight probe."""
    calls = {"get": 0, "head": 0}

    class _Resp:
        def __init__(self, payload=None):
            self._payload = payload or {}

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class _Client:
        async def get(self, *args, **kwargs):
            calls["get"] += 1
            return _Resp({"models": [{"name": "qwen3.5:2b"}]})

        async def head(self, *args, **kwargs):
            calls["head"] += 1
            return _Resp()

    async def _client():
        return _Client()

    pa.reset_ollama_tags_cache()
    monkeypatch.setattr(pa, "get_http_client", _client)
    tags_first = await pa.fetch_ollama_tags(force_refresh=True, ttl_seconds=60)
    tags_second = await pa.fetch_ollama_tags(force_refresh=False, ttl_seconds=60)
    await pa.probe_ollama_liveness()

    assert tags_first == tags_second
    assert calls["get"] == 1
    assert calls["head"] == 1
