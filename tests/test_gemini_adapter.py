# Objective: Test coverage for the Gemini SDK adapter (google-genai and legacy generativeai paths).
"""providers._gemini.GeminiProvider.GeminiAdapter builds the right call for each SDK."""

import base64
import sys
from types import SimpleNamespace

import app.providers_async  # noqa: F401  carrega o pacote de providers (importar _gemini antes cai num ciclo)
import pytest

_gemini = sys.modules["app.providers._gemini"]


class _Recorder:
    def __init__(self, response):
        self.calls = []
        self.response = response

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


@pytest.fixture
def genai_client(monkeypatch):
    generate = _Recorder(SimpleNamespace(text="resposta"))
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    monkeypatch.setattr(_gemini, "google_genai", SimpleNamespace(Client=lambda api_key=None: client))
    monkeypatch.setattr(
        _gemini.ptools, "to_gemini_response_config", lambda fmt: {"response_mime_type": "x"} if fmt else {}
    )
    return generate


@pytest.fixture
def legacy_sdk(monkeypatch):
    models = []

    class _Model:
        def __init__(self, name, **kwargs):
            self.name, self.kwargs, self.calls = name, kwargs, []
            models.append(self)

        def generate_content(self, contents, **kwargs):
            self.calls.append((contents, kwargs))
            return "raw"

    monkeypatch.setattr(_gemini, "google_genai", None)
    monkeypatch.setattr(_gemini._pa, "genai", SimpleNamespace(GenerativeModel=_Model))
    monkeypatch.setattr(_gemini.ptools, "to_gemini_response_config", lambda fmt: {})
    return models


def _generate(**overrides):
    kwargs = {"model_name": "gemini-x", "prompt": "oi", "image_b64": None, "temperature": 0.2, "max_tokens": 64}
    return _gemini.GeminiProvider.GeminiAdapter().generate(**{**kwargs, **overrides})


def test_genai_plain_text_returns_text_namespace(genai_client):
    out = _generate(image_b64="aW1n", system_instruction="seja breve")
    assert out.text == "resposta"
    ((_, kwargs),) = genai_client.calls
    assert kwargs["model"] == "gemini-x"
    assert kwargs["contents"] == [{"text": "oi"}, {"inline_data": {"mime_type": "image/jpeg", "data": "aW1n"}}]
    assert kwargs["config"] == {
        "temperature": 0.2,
        "max_output_tokens": 64 + 4096,  # o raciocínio conta dentro do teto de saída
        "thinking_config": {"thinking_budget": 4096},
        "system_instruction": "seja breve",
    }


def test_genai_with_tools_returns_raw_response(genai_client):
    contents = [{"role": "user", "parts": [{"text": "q"}]}]
    out = _generate(
        tools=[{"fn": 1}], tool_config={"mode": "AUTO"}, contents=contents, response_format={"type": "json"}
    )
    assert out is genai_client.response
    ((_, kwargs),) = genai_client.calls
    assert kwargs["contents"] is contents
    assert kwargs["config"]["tools"] == [{"fn": 1}] and kwargs["config"]["tool_config"] == {"mode": "AUTO"}
    assert kwargs["config"]["response_mime_type"] == "x"


def test_legacy_sdk_decodes_image_and_places_options(legacy_sdk):
    image = base64.b64encode(b"jpeg").decode()
    assert _generate(image_b64=image, tools=["t"], tool_config={"m": 1}, system_instruction="s") == "raw"
    (model,) = legacy_sdk
    assert model.name == "gemini-x" and model.kwargs == {"system_instruction": "s", "tools": ["t"]}
    contents, kwargs = model.calls[0]
    assert contents == ["oi", {"mime_type": "image/jpeg", "data": b"jpeg"}]
    assert kwargs == {"generation_config": {"temperature": 0.2, "max_output_tokens": 64}, "tool_config": {"m": 1}}


def test_legacy_sdk_passes_contents_through(legacy_sdk):
    _generate(contents=["historico"])
    assert legacy_sdk[0].calls[0][0] == ["historico"]
    assert legacy_sdk[0].kwargs == {}


def test_no_sdk_available(monkeypatch):
    monkeypatch.setattr(_gemini, "google_genai", None)
    monkeypatch.setattr(_gemini._pa, "genai", None)
    with pytest.raises(ImportError):
        _generate()
