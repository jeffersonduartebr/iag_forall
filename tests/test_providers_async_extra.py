"""Módulo `tests/test_providers_async_extra.py`: descreve responsabilidades e integrações deste arquivo."""

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
    monkeypatch.setattr(pa.ProviderFactory, "_instances", {})
    assert pa.ProviderFactory.get_provider("openai/gpt-4o") == "openai-provider"
    assert pa.ProviderFactory.get_provider("ollama/phi4:latest") == "ollama-provider"

    with pytest.raises(ValueError):
        pa.ProviderFactory.get_provider("xpto/model")


@pytest.mark.asyncio
async def test_call_model_success_and_error_categories(monkeypatch):
    """Testa call model success and error categories."""
    class _P:
        """Classe `_P`: concentra responsabilidades de test providers async extra."""
        async def generate(self, **kwargs):
            """Executa generate."""
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
        """Classe `_PCircuit`: concentra responsabilidades de test providers async extra."""
        async def generate(self, **kwargs):
            """Executa generate."""
            raise pybreaker.CircuitBreakerError("open")

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", lambda m: _PCircuit())
    with pytest.raises(pa.ProviderCircuitOpenError):
        await pa.call_model("openai/gpt-4o", "x")

    class _PTimeout:
        """Classe `_PTimeout`: concentra responsabilidades de test providers async extra."""
        async def generate(self, **kwargs):
            """Executa generate."""
            raise asyncio.TimeoutError("timeout")

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", lambda m: _PTimeout())
    with pytest.raises(pa.ProviderCallError) as exc1:
        await pa.call_model("openai/gpt-4o", "x")
    assert exc1.value.category == "provider_timeout"

    class RateLimitError(Exception):
        """Classe `RateLimitError`: concentra responsabilidades de test providers async extra."""
        pass

    class _PRL:
        """Classe `_PRL`: concentra responsabilidades de test providers async extra."""
        async def generate(self, **kwargs):
            """Executa generate."""
            raise RateLimitError("rl")

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", lambda m: _PRL())
    with pytest.raises(pa.ProviderCallError) as exc2:
        await pa.call_model("openai/gpt-4o", "x")
    assert exc2.value.category == "provider_rate_limit"


@pytest.mark.asyncio
async def test_ensure_ollama_model_async_paths(monkeypatch):
    """Testa ensure ollama model async paths."""
    class _Resp:
        """Classe `_Resp`: concentra responsabilidades de test providers async extra."""
        def __init__(self, code, payload=None):
            """Inicializa estado interno necessário para uso da classe."""
            self.status_code = code
            self._payload = payload or {}

        def json(self):
            """Executa json."""
            return self._payload

    class _Client:
        """Classe `_Client`: concentra responsabilidades de test providers async extra."""
        def __init__(self, tags_resp, pull_resp):
            """Inicializa estado interno necessário para uso da classe."""
            self.tags_resp = tags_resp
            self.pull_resp = pull_resp

        async def get(self, *a, **k):
            """Executa get."""
            return self.tags_resp

        async def post(self, *a, **k):
            """Executa post."""
            return self.pull_resp

    async def _c1():
        """Executa c1."""
        return _Client(_Resp(200, {"models": [{"name": "phi4:latest"}]}), _Resp(200))

    monkeypatch.setattr(pa, "get_http_client", _c1)
    assert await pa._ensure_ollama_model_async("phi4:latest") is True

    async def _c2():
        """Executa c2."""
        return _Client(_Resp(200, {"models": []}), _Resp(200))

    monkeypatch.setattr(pa, "get_http_client", _c2)
    assert await pa._ensure_ollama_model_async("phi4:latest") is True

    async def _c3():
        """Executa c3."""
        return _Client(_Resp(500, {}), _Resp(500))

    monkeypatch.setattr(pa, "get_http_client", _c3)
    assert await pa._ensure_ollama_model_async("phi4:latest") is False

    async def _boom():
        """Executa boom."""
        raise httpx.ConnectError("x")

    monkeypatch.setattr(pa, "get_http_client", _boom)
    assert await pa._ensure_ollama_model_async("phi4:latest") is False
