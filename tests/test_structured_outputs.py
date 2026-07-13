# Objective: Unit tests for structured-output (response_format) translation helpers.
"""Testes das funções puras de tradução de response_format (JSON mode) por provider."""

from app import provider_tools as ptools

JSON_OBJECT = {"type": "json_object"}
TEXT = {"type": "text"}
SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
    "title": "Weather",
}
JSON_SCHEMA = {"type": "json_schema", "json_schema": {"name": "weather", "schema": SCHEMA, "strict": True}}


# --------------------------------------------------------------------------
# wants_json
# --------------------------------------------------------------------------
def test_wants_json_true_for_json_variants():
    assert ptools.wants_json(JSON_OBJECT) is True
    assert ptools.wants_json(JSON_SCHEMA) is True


def test_wants_json_false_for_text_and_none():
    assert ptools.wants_json(TEXT) is False
    assert ptools.wants_json(None) is False
    assert ptools.wants_json({}) is False
    assert ptools.wants_json("json") is False


# --------------------------------------------------------------------------
# to_ollama_format
# --------------------------------------------------------------------------
def test_to_ollama_format_json_object():
    assert ptools.to_ollama_format(JSON_OBJECT) == "json"


def test_to_ollama_format_json_schema_returns_schema_dict():
    assert ptools.to_ollama_format(JSON_SCHEMA) == SCHEMA


def test_to_ollama_format_text_and_none():
    assert ptools.to_ollama_format(TEXT) is None
    assert ptools.to_ollama_format(None) is None


def test_to_ollama_format_json_schema_without_schema_falls_back_to_json():
    assert ptools.to_ollama_format({"type": "json_schema", "json_schema": {"name": "x"}}) == "json"


# --------------------------------------------------------------------------
# to_gemini_response_config
# --------------------------------------------------------------------------
def test_to_gemini_response_config_json_object():
    cfg = ptools.to_gemini_response_config(JSON_OBJECT)
    assert cfg == {"response_mime_type": "application/json"}


def test_to_gemini_response_config_json_schema_cleans_unsupported_keys():
    cfg = ptools.to_gemini_response_config(JSON_SCHEMA)
    assert cfg["response_mime_type"] == "application/json"
    schema = cfg["response_schema"]
    # chaves não aceitas pelo Gemini são removidas (additionalProperties, title)
    assert "additionalProperties" not in schema
    assert "title" not in schema
    assert schema["properties"]["city"] == {"type": "string"}


def test_to_gemini_response_config_text_and_none():
    assert ptools.to_gemini_response_config(TEXT) == {}
    assert ptools.to_gemini_response_config(None) == {}


# --------------------------------------------------------------------------
# json_mode_system_suffix
# --------------------------------------------------------------------------
def test_json_mode_system_suffix_json_object():
    suffix = ptools.json_mode_system_suffix(JSON_OBJECT)
    assert suffix is not None
    assert "JSON" in suffix


def test_json_mode_system_suffix_json_schema_embeds_schema():
    suffix = ptools.json_mode_system_suffix(JSON_SCHEMA)
    assert suffix is not None
    assert "city" in suffix  # o schema serializado é anexado à instrução


def test_json_mode_system_suffix_text_and_none():
    assert ptools.json_mode_system_suffix(TEXT) is None
    assert ptools.json_mode_system_suffix(None) is None
