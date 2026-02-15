"""Módulo `tests/test_providers_async_core_reliability.py`: descreve responsabilidades e integrações deste arquivo."""

import asyncio
from types import SimpleNamespace

import pybreaker
import pytest
import requests

from app import providers_async as pa


def _close_breakers():
    """Executa close breakers."""
    pa.cloud_breaker._state = pybreaker.CircuitClosedState(pa.cloud_breaker)
    pa.local_breaker._state = pybreaker.CircuitClosedState(pa.local_breaker)


@pytest.mark.asyncio
async def test_openai_provider_generate_success_with_reasoning_model(monkeypatch):
    """Testa openai provider generate success with reasoning model."""
    _close_breakers()

    class _Create:
        """Classe `_Create`: concentra responsabilidades de test providers async core reliability."""
        def __init__(self):
            """Inicializa estado interno necessário para uso da classe."""
            self.calls = []

        async def create(self, **kwargs):
            """Executa create."""
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok-openai"))],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=22),
                model_dump=lambda: {"id": "x"},
            )

    create = _Create()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create.create)))

    monkeypatch.setattr(pa, "AsyncOpenAI", lambda api_key=None: fake_client)
    monkeypatch.setattr(pa, "get_model_cost", lambda model, p, c: 0.123)

    provider = pa.OpenAIProvider()
    out = await provider.generate(prompt="hello", image_b64="abc", model="gpt-5-mini", max_tokens=64)

    assert out.text == "ok-openai"
    assert out.prompt_tokens == 11
    assert out.completion_tokens == 22
    assert out.cost == 0.123
    assert "max_completion_tokens" in create.calls[0]
    assert "max_tokens" not in create.calls[0]


@pytest.mark.asyncio
async def test_anthropic_provider_generate_success(monkeypatch):
    """Testa anthropic provider generate success."""
    _close_breakers()

    class _Messages:
        """Classe `_Messages`: concentra responsabilidades de test providers async core reliability."""
        async def create(self, **kwargs):
            """Executa create."""
            return SimpleNamespace(
                content=[SimpleNamespace(text="ok-anthropic")],
                usage=SimpleNamespace(input_tokens=7, output_tokens=9),
            )

    monkeypatch.setattr(pa, "AsyncAnthropic", lambda api_key=None: SimpleNamespace(messages=_Messages()))
    monkeypatch.setattr(pa, "get_model_cost", lambda model, p, c: 0.5)

    provider = pa.AnthropicProvider()
    out = await provider.generate(prompt="hello", model="claude-3-5-sonnet-latest")

    assert out.text == "ok-anthropic"
    assert out.prompt_tokens == 7
    assert out.completion_tokens == 9
    assert out.cost == 0.5


@pytest.mark.asyncio
async def test_gemini_provider_generate_success(monkeypatch):
    """Testa gemini provider generate success."""
    _close_breakers()

    class _GModel:
        """Classe `_GModel`: concentra responsabilidades de test providers async core reliability."""
        def __init__(self, name):
            """Inicializa estado interno necessário para uso da classe."""
            self.name = name

        def generate_content(self, parts, generation_config=None):
            """Executa generate content."""
            return SimpleNamespace(text="ok-gemini")

    monkeypatch.setattr(pa, "genai", SimpleNamespace(GenerativeModel=_GModel))
    monkeypatch.setattr(pa, "count_tokens", lambda text, model: len(text))
    monkeypatch.setattr(pa, "get_model_cost", lambda model, p, c: 0.02)

    provider = pa.GeminiProvider()
    out = await provider.generate(prompt="hello", model="gemini-1.5-flash")

    assert out.text == "ok-gemini"
    assert out.prompt_tokens == 5
    assert out.completion_tokens == 9
    assert out.cost == 0.02


@pytest.mark.asyncio
async def test_close_http_client_and_render_metrics(monkeypatch):
    """Testa close http client and render metrics."""
    class _Client:
        """Classe `_Client`: concentra responsabilidades de test providers async core reliability."""
        def __init__(self):
            """Inicializa estado interno necessário para uso da classe."""
            self.is_closed = False
            self.closed = False

        async def close(self):
            """Executa close."""
            self.closed = True
            self.is_closed = True

    c = _Client()
    monkeypatch.setattr(pa, "_http_client", c)
    await pa.close_http_client()
    assert c.closed is True
    data, ctype = pa.render_metrics_response()
    assert data is not None
    assert ctype


def test_ensure_ollama_model_sync_paths(monkeypatch):
    """Testa ensure ollama model sync paths."""
    class _Resp:
        """Classe `_Resp`: concentra responsabilidades de test providers async core reliability."""
        def __init__(self, code=200, payload=None):
            """Inicializa estado interno necessário para uso da classe."""
            self.status_code = code
            self._payload = payload or {}

        def json(self):
            """Executa json."""
            return self._payload

        def iter_lines(self):
            """Executa iter lines."""
            yield b"line"

    # already installed
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(200, {"models": [{"name": "phi4:latest"}]}))
    assert pa._ensure_ollama_model("phi4:latest") is True

    # pull success
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(200, {"models": []}))
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(200))
    assert pa._ensure_ollama_model("phi4:latest") is True

    # pull fail status
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(500))
    assert pa._ensure_ollama_model("phi4:latest") is False

    # exception branches
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.Timeout()))
    assert pa._ensure_ollama_model("phi4:latest") is False

    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.ConnectionError("x")))
    assert pa._ensure_ollama_model("phi4:latest") is False

    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert pa._ensure_ollama_model("phi4:latest") is False


@pytest.mark.asyncio
async def test_call_model_error_categories_timeout_and_unavailable(monkeypatch):
    """Testa call model error categories timeout and unavailable."""
    class _PTimeout:
        """Classe `_PTimeout`: concentra responsabilidades de test providers async core reliability."""
        async def generate(self, **kwargs):
            """Executa generate."""
            raise asyncio.TimeoutError("timeout")

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", lambda model: _PTimeout())
    with pytest.raises(pa.ProviderCallError) as exc1:
        await pa.call_model("openai/gpt-4o", "q")
    assert exc1.value.category == "provider_timeout"

    class _PUnavailable:
        """Classe `_PUnavailable`: concentra responsabilidades de test providers async core reliability."""
        async def generate(self, **kwargs):
            """Executa generate."""
            raise RuntimeError("down")

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", lambda model: _PUnavailable())
    with pytest.raises(pa.ProviderCallError) as exc2:
        await pa.call_model("openai/gpt-4o", "q")
    assert exc2.value.category == "provider_unavailable"
