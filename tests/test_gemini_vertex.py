# Objective: Gemini through Vertex AI (billed to the Caso 1 GCP project) and billed-token accounting.
"""GEMINI_VERTEX_PROJECT routes gemini/* to Vertex AI with the environment's credentials; token counts (and so
cost, one of the three routing objectives) come from the provider's usage metadata, thinking tokens included."""

from types import SimpleNamespace

import app.providers_async  # noqa: F401  (carrega os provedores na ordem certa)
import pytest
from app.model_registry import is_provider_configured
from app.providers import _gemini
from app.utils.pricing import _lookup_fallback


class _Sdk:
    def __init__(self):
        self.criados = []

    def Client(self, **kwargs):  # noqa: N802 (espelha a API do SDK)
        self.criados.append(kwargs)
        return SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(text="ok", kw=kw)))


@pytest.fixture
def sdk(monkeypatch):
    fake = _Sdk()
    monkeypatch.setattr(_gemini, "google_genai", fake)
    _gemini._CLIENTES.clear()
    yield fake
    _gemini._CLIENTES.clear()


def test_vertex_client_when_project_is_configured_and_reused(sdk, monkeypatch):
    monkeypatch.setattr(_gemini, "GEMINI_VERTEX_PROJECT", "aristo-caso1")
    monkeypatch.setattr(_gemini, "GEMINI_VERTEX_LOCATION", "global")
    assert _gemini._cliente_genai() is _gemini._cliente_genai()
    assert sdk.criados == [{"vertexai": True, "project": "aristo-caso1", "location": "global"}]


def test_api_key_client_otherwise(sdk, monkeypatch):
    monkeypatch.setattr(_gemini, "GEMINI_VERTEX_PROJECT", "")
    monkeypatch.setattr(_gemini._pa, "GEMINI_API_KEY", "k")
    _gemini._cliente_genai()
    assert sdk.criados == [{"api_key": "k"}]


def test_generation_goes_through_the_cached_client(sdk, monkeypatch):
    monkeypatch.setattr(_gemini, "GEMINI_VERTEX_PROJECT", "p")
    options = _gemini._GenOptions(0.2, 900, None, None, "sistema", None)
    resposta = _gemini.GeminiProvider.GeminiAdapter._generate_genai("gemini-3.8-flash", [{"text": "oi"}], options)
    assert resposta.kw["model"] == "gemini-3.8-flash" and resposta.kw["config"]["system_instruction"] == "sistema"


def test_legacy_sdk_refuses_vertex(monkeypatch):
    monkeypatch.setattr(_gemini, "google_genai", None)
    monkeypatch.setattr(_gemini, "GEMINI_VERTEX_PROJECT", "p")
    monkeypatch.setattr(_gemini._pa, "genai", SimpleNamespace())
    with pytest.raises(ImportError, match="Vertex"):
        _gemini.GeminiProvider.GeminiAdapter().generate("gemini-3.8-flash", "oi", None, 0.2, 100)


def test_billed_tokens_include_thinking():
    uso = SimpleNamespace(prompt_token_count=8, candidates_token_count=1, thoughts_token_count=129)
    assert _gemini._tokens_cobrados(SimpleNamespace(usage_metadata=uso)) == (8, 130)
    assert _gemini._tokens_cobrados(SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=None))) == (3, 0)
    assert _gemini._tokens_cobrados(SimpleNamespace(text="x")) is None


def test_provider_is_configured_by_vertex_project_alone(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_VERTEX_PROJECT", "aristo-caso1")
    assert is_provider_configured("gemini")


def test_official_vertex_price():
    assert _lookup_fallback("gemini/gemini-3.8-flash") == {"in": 0.00075, "out": 0.00375}


def test_concurrent_first_calls_build_a_single_client(sdk, monkeypatch):
    """Production 2026-09-25: parallel judge calls each built a client; the replaced ones were collected and closed
    their HTTP connection under a thread still using them ("the client has been closed")."""
    import threading
    import time

    monkeypatch.setattr(_gemini, "GEMINI_VERTEX_PROJECT", "aristo-caso1")
    original = sdk.Client

    def _lento(**kw):
        time.sleep(0.05)  # alarga a janela da corrida
        return original(**kw)

    monkeypatch.setattr(sdk, "Client", _lento)
    obtidos = []
    threads = [threading.Thread(target=lambda: obtidos.append(_gemini._cliente_genai())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(sdk.criados) == 1 and len({id(c) for c in obtidos}) == 1
