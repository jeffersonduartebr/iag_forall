# Objective: Unit tests for provider-native (server-side) tool classification/routing.
"""Testes das funções puras de classificação/roteamento de tools nativas."""

from app import native_tools as nt

FUNCTION_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
}
ANTHROPIC_WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search"}
ANTHROPIC_BASH = {"type": "bash_20250124", "name": "bash"}
ANTHROPIC_TEXT_EDITOR = {"type": "text_editor_20250124", "name": "str_replace_editor"}
ANTHROPIC_COMPUTER = {"type": "computer_20250124", "name": "computer"}
GEMINI_SEARCH = {"google_search": {}}
GEMINI_SEARCH_RETRIEVAL = {"google_search_retrieval": {}}
GEMINI_CODE_EXEC = {"code_execution": {}}
OPENAI_WEB_SEARCH = {"type": "web_search"}
OPENAI_CODE_INTERPRETER = {"type": "code_interpreter"}
OPENAI_FILE_SEARCH = {"type": "file_search"}


# --------------------------------------------------------------------------
# is_native_tool
# --------------------------------------------------------------------------
def test_is_native_tool_function_is_not_native():
    assert nt.is_native_tool(FUNCTION_TOOL) is False
    assert nt.is_native_tool({"type": "function"}) is False
    assert nt.is_native_tool({"type": "FUNCTION"}) is False  # case-insensitive


def test_is_native_tool_typed_server_tools():
    assert nt.is_native_tool(ANTHROPIC_WEB_SEARCH) is True
    assert nt.is_native_tool(OPENAI_CODE_INTERPRETER) is True
    assert nt.is_native_tool({"type": "anything_else"}) is True


def test_is_native_tool_gemini_grounding_keys():
    assert nt.is_native_tool(GEMINI_SEARCH) is True
    assert nt.is_native_tool(GEMINI_SEARCH_RETRIEVAL) is True
    assert nt.is_native_tool(GEMINI_CODE_EXEC) is True


def test_is_native_tool_rejects_non_dicts_and_empty():
    assert nt.is_native_tool(None) is False
    assert nt.is_native_tool("web_search") is False
    assert nt.is_native_tool(["web_search"]) is False
    assert nt.is_native_tool({}) is False
    assert nt.is_native_tool({"foo": "bar"}) is False  # sem type e sem chave de grounding


# --------------------------------------------------------------------------
# native_tool_type
# --------------------------------------------------------------------------
def test_native_tool_type_returns_versioned_type():
    assert nt.native_tool_type(ANTHROPIC_WEB_SEARCH) == "web_search_20250305"
    assert nt.native_tool_type(OPENAI_WEB_SEARCH) == "web_search"


def test_native_tool_type_returns_grounding_key():
    assert nt.native_tool_type(GEMINI_SEARCH) == "google_search"
    assert nt.native_tool_type(GEMINI_CODE_EXEC) == "code_execution"


def test_native_tool_type_empty_for_non_native():
    assert nt.native_tool_type(FUNCTION_TOOL) == ""
    assert nt.native_tool_type({"type": "function"}) == ""
    assert nt.native_tool_type(None) == ""
    assert nt.native_tool_type({"foo": "bar"}) == ""


# --------------------------------------------------------------------------
# split_tools
# --------------------------------------------------------------------------
def test_split_tools_partitions_and_preserves_order():
    tools = [FUNCTION_TOOL, GEMINI_SEARCH, ANTHROPIC_WEB_SEARCH, FUNCTION_TOOL]
    functions, natives = nt.split_tools(tools)
    assert functions == [FUNCTION_TOOL, FUNCTION_TOOL]
    assert natives == [GEMINI_SEARCH, ANTHROPIC_WEB_SEARCH]


def test_split_tools_handles_none_and_empty():
    assert nt.split_tools(None) == ([], [])
    assert nt.split_tools([]) == ([], [])


def test_split_tools_all_function_or_all_native():
    assert nt.split_tools([FUNCTION_TOOL]) == ([FUNCTION_TOOL], [])
    assert nt.split_tools([OPENAI_WEB_SEARCH]) == ([], [OPENAI_WEB_SEARCH])


# --------------------------------------------------------------------------
# NATIVE_TOOL_SUPPORT mapping
# --------------------------------------------------------------------------
def test_native_tool_support_map_shape():
    for family in ("anthropic", "claude", "gemini", "openai", "openrouter", "ollama"):
        assert family in nt.NATIVE_TOOL_SUPPORT
    # Ollama não suporta nenhuma tool nativa.
    assert nt.NATIVE_TOOL_SUPPORT["ollama"] == frozenset()
    # anthropic e claude apontam para o mesmo conjunto.
    assert nt.NATIVE_TOOL_SUPPORT["anthropic"] == nt.NATIVE_TOOL_SUPPORT["claude"]
    assert {"web_search", "bash", "text_editor", "computer"} <= set(nt.NATIVE_TOOL_SUPPORT["anthropic"])
    assert {"google_search", "code_execution"} <= set(nt.NATIVE_TOOL_SUPPORT["gemini"])
    assert {"web_search", "code_interpreter", "file_search"} <= set(nt.NATIVE_TOOL_SUPPORT["openai"])


# --------------------------------------------------------------------------
# provider_family
# --------------------------------------------------------------------------
def test_provider_family_from_prefix():
    assert nt.provider_family("openai/gpt-4o") == "openai"
    assert nt.provider_family("anthropic/claude-3-5-sonnet-latest") == "anthropic"
    assert nt.provider_family("gemini/gemini-1.5-pro") == "gemini"
    assert nt.provider_family("ollama/phi4:latest") == "ollama"
    assert nt.provider_family("openrouter/meta/llama-3.1-70b") == "openrouter"


def test_provider_family_from_registry_short_name():
    # Nome curto registrado no registry resolve pelo provedor autoritativo.
    assert nt.provider_family("claude-3-5-sonnet-latest") == "anthropic"
    assert nt.provider_family("gpt-4o") == "openai"


def test_provider_family_marker_fallback():
    # Modelos desconhecidos caem para marcadores no nome.
    assert nt.provider_family("weird-claude-model") == "anthropic"
    assert nt.provider_family("google-experimental") == "gemini"
    assert nt.provider_family("o3-mini-preview") == "openai"


def test_provider_family_invalid_input():
    assert nt.provider_family("") == ""
    assert nt.provider_family(None) == ""  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# provider_supports_native_tools (prefix matching + família)
# --------------------------------------------------------------------------
def test_supports_no_native_tools_is_always_true():
    assert nt.provider_supports_native_tools("ollama/phi4:latest", []) is True
    assert nt.provider_supports_native_tools("ollama/phi4:latest", None) is True
    # Uma function tool não é nativa: nada a exigir.
    assert nt.provider_supports_native_tools("ollama/phi4:latest", [FUNCTION_TOOL]) is True


def test_supports_prefix_matching_versioned_types():
    # web_search_20250305 casa o prefixo web_search do Anthropic.
    assert nt.provider_supports_native_tools("anthropic/claude-3-5-sonnet-latest", [ANTHROPIC_WEB_SEARCH]) is True
    assert nt.provider_supports_native_tools("anthropic/claude-3-5-sonnet-latest", [ANTHROPIC_BASH]) is True
    assert nt.provider_supports_native_tools(
        "anthropic/claude-3-5-sonnet-latest", [ANTHROPIC_TEXT_EDITOR, ANTHROPIC_COMPUTER]
    ) is True


def test_supports_gemini_and_openai_families():
    assert nt.provider_supports_native_tools("gemini/gemini-1.5-pro", [GEMINI_SEARCH, GEMINI_CODE_EXEC]) is True
    assert nt.provider_supports_native_tools("openai/gpt-4o", [OPENAI_CODE_INTERPRETER, OPENAI_FILE_SEARCH]) is True


def test_supports_rejects_cross_provider_native_tools():
    # Gemini não suporta a web_search do Anthropic/OpenAI.
    assert nt.provider_supports_native_tools("gemini/gemini-1.5-pro", [ANTHROPIC_WEB_SEARCH]) is False
    # Anthropic não suporta grounding do Gemini.
    assert nt.provider_supports_native_tools("anthropic/claude-3-5-sonnet-latest", [GEMINI_SEARCH]) is False
    # Ollama não suporta nenhuma tool nativa.
    assert nt.provider_supports_native_tools("ollama/phi4:latest", [OPENAI_WEB_SEARCH]) is False


def test_supports_requires_all_natives_supported():
    # Um suportado + um não suportado → False (exige TODAS).
    assert nt.provider_supports_native_tools(
        "openai/gpt-4o", [OPENAI_WEB_SEARCH, GEMINI_SEARCH]
    ) is False


# --------------------------------------------------------------------------
# filter_native_tool_capable_model_names
# --------------------------------------------------------------------------
MODELS = [
    "anthropic/claude-3-5-sonnet-latest",
    "openai/gpt-4o",
    "gemini/gemini-1.5-pro",
    "ollama/phi4:latest",
    "openrouter/meta/llama-3.1-70b",
]


def test_filter_web_search_keeps_supporting_providers():
    kept = nt.filter_native_tool_capable_model_names(MODELS, [ANTHROPIC_WEB_SEARCH])
    # web_search é suportado por anthropic, openai e openrouter; não por gemini/ollama.
    assert set(kept) == {
        "anthropic/claude-3-5-sonnet-latest",
        "openai/gpt-4o",
        "openrouter/meta/llama-3.1-70b",
    }


def test_filter_grounding_keeps_only_gemini():
    kept = nt.filter_native_tool_capable_model_names(MODELS, [GEMINI_SEARCH])
    assert kept == ["gemini/gemini-1.5-pro"]


def test_filter_no_native_tools_returns_all_strings():
    assert nt.filter_native_tool_capable_model_names(MODELS, []) == MODELS
    assert nt.filter_native_tool_capable_model_names(MODELS, [FUNCTION_TOOL]) == MODELS


def test_filter_drops_non_string_entries_when_native_present():
    mixed = ["openai/gpt-4o", None, 123, "ollama/phi4:latest"]
    kept = nt.filter_native_tool_capable_model_names(mixed, [OPENAI_WEB_SEARCH])
    assert kept == ["openai/gpt-4o"]


def test_filter_empty_when_no_provider_supports():
    kept = nt.filter_native_tool_capable_model_names(["ollama/phi4:latest"], [OPENAI_WEB_SEARCH])
    assert kept == []
