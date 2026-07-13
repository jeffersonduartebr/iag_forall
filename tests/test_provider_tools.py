# Objective: Unit tests for the canonical<->provider tool-calling translation layer.
"""Testes das funções puras de tradução de tool/function calling."""

import json
from types import SimpleNamespace

from app import native_tools as nt
from app import provider_tools as pt

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "clima atual",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]

# Tools nativas (server-side) por provedor, no array canônico ``tools``.
ANTHROPIC_SERVER_TOOL = {"type": "web_search_20250305", "name": "web_search"}
GEMINI_GROUNDING_TOOL = {"google_search": {}}
GEMINI_CODE_EXEC_TOOL = {"code_execution": {}}
OPENAI_HOSTED_TOOL = {"type": "web_search"}

MULTITURN = [
    {"role": "user", "content": "Qual o clima em Natal?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Natal"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "32C, ensolarado"},
]


# --------------------------------------------------------------------------
# build_provider_messages
# --------------------------------------------------------------------------
def test_build_messages_single_turn_text():
    assert pt.build_provider_messages("olá", None, None, None) == [{"role": "user", "content": "olá"}]


def test_build_messages_single_turn_with_system_and_image():
    msgs = pt.build_provider_messages("descreva", "seja breve", None, "BASE64IMG")
    assert msgs[0] == {"role": "system", "content": "seja breve"}
    assert msgs[1]["role"] == "user"
    parts = msgs[1]["content"]
    assert parts[0] == {"type": "text", "text": "descreva"}
    assert parts[1]["type"] == "image_url" and "BASE64IMG" in parts[1]["image_url"]["url"]


def test_build_messages_multiturn_is_source_of_truth():
    msgs = pt.build_provider_messages("ignored", None, MULTITURN, None)
    assert msgs == MULTITURN
    assert msgs is not MULTITURN  # cópia rasa, não a mesma referência


def test_build_messages_multiturn_prepends_system_when_absent():
    msgs = pt.build_provider_messages("x", "sys", MULTITURN, None)
    assert msgs[0] == {"role": "system", "content": "sys"}


# --------------------------------------------------------------------------
# canonical helpers
# --------------------------------------------------------------------------
def test_canonical_tool_call_normalizes_args_to_json_string():
    tc = pt.canonical_tool_call("f", {"a": 1}, call_id="x")
    assert tc == {"id": "x", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}


def test_canonical_tool_call_synthesizes_id_when_missing():
    tc = pt.canonical_tool_call("f", "{}")
    assert tc["id"] and tc["id"].startswith("call_")


def test_tools_disabled():
    assert pt.tools_disabled("none") is True
    assert pt.tools_disabled("auto") is False
    assert pt.tools_disabled(None) is False
    assert pt.tools_disabled({"type": "function"}) is False


# --------------------------------------------------------------------------
# OpenAI serialization (SDK-like objects)
# --------------------------------------------------------------------------
def test_serialize_openai_tool_calls_from_sdk_objects():
    raw = [
        SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(name="get_weather", arguments='{"city":"Natal"}'),
        )
    ]
    out = pt.serialize_openai_tool_calls(raw)
    assert out == [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Natal"}'}}
    ]


def test_serialize_openai_tool_calls_none():
    assert pt.serialize_openai_tool_calls(None) is None
    assert pt.serialize_openai_tool_calls([]) is None


def test_openai_finish_reason_prefers_tool_calls():
    assert pt.openai_finish_reason("stop", True) == "tool_calls"
    assert pt.openai_finish_reason("length", False) == "length"
    assert pt.openai_finish_reason(None, False) == "stop"


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------
def test_to_anthropic_tools_uses_input_schema():
    out = pt.to_anthropic_tools(TOOLS)
    assert out[0]["name"] == "get_weather"
    assert "input_schema" in out[0] and out[0]["input_schema"]["properties"]["city"]["type"] == "string"


def test_to_anthropic_tool_choice_mapping():
    assert pt.to_anthropic_tool_choice("auto") == {"type": "auto"}
    assert pt.to_anthropic_tool_choice("required") == {"type": "any"}
    assert pt.to_anthropic_tool_choice({"type": "function", "function": {"name": "f"}}) == {"type": "tool", "name": "f"}
    assert pt.to_anthropic_tool_choice("none") is None


def test_to_anthropic_messages_translates_tool_turns():
    system, msgs = pt.to_anthropic_messages(MULTITURN)
    assert system is None
    assert msgs[0] == {"role": "user", "content": [{"type": "text", "text": "Qual o clima em Natal?"}]}
    assert msgs[1]["role"] == "assistant"
    tool_use = msgs[1]["content"][0]
    assert tool_use["type"] == "tool_use" and tool_use["name"] == "get_weather" and tool_use["input"] == {"city": "Natal"}
    assert msgs[2]["role"] == "user"
    tool_result = msgs[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "c1"


def test_from_anthropic_response_text_and_tool_use():
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="deixa eu ver "),
            SimpleNamespace(type="tool_use", id="tu1", name="get_weather", input={"city": "Natal"}),
        ],
        stop_reason="tool_use",
    )
    text, tool_calls, finish = pt.from_anthropic_response(resp)
    assert text == "deixa eu ver "
    assert finish == "tool_calls"
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Natal"}


def test_from_anthropic_response_plain_text():
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="oi")], stop_reason="end_turn")
    text, tool_calls, finish = pt.from_anthropic_response(resp)
    assert text == "oi" and tool_calls is None and finish == "stop"


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------
def test_to_gemini_tools_and_schema_cleanup():
    out = pt.to_gemini_tools(TOOLS)
    decls = out[0]["function_declarations"]
    assert decls[0]["name"] == "get_weather"
    # additionalProperties deve ser removido (não aceito pelo Gemini)
    assert "additionalProperties" not in decls[0]["parameters"]


def test_to_gemini_tool_config_modes():
    assert pt.to_gemini_tool_config("auto")["function_calling_config"]["mode"] == "AUTO"
    assert pt.to_gemini_tool_config("required")["function_calling_config"]["mode"] == "ANY"
    assert pt.to_gemini_tool_config("none")["function_calling_config"]["mode"] == "NONE"
    named = pt.to_gemini_tool_config({"type": "function", "function": {"name": "f"}})
    assert named["function_calling_config"]["allowed_function_names"] == ["f"]


def test_to_gemini_contents_translates_tool_turns():
    system, contents = pt.to_gemini_contents(MULTITURN)
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["function_call"]["name"] == "get_weather"
    fr = contents[2]["parts"][0]["function_response"]
    assert fr["name"] == "get_weather" and fr["response"] == {"result": "32C, ensolarado"}


def test_from_gemini_response_function_call():
    part = SimpleNamespace(function_call=SimpleNamespace(name="get_weather", args={"city": "Natal"}), text=None)
    resp = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
    text, tool_calls, finish = pt.from_gemini_response(resp)
    assert finish == "tool_calls"
    assert tool_calls[0]["function"]["name"] == "get_weather"


def test_from_gemini_response_simplenamespace_text_fallback():
    # Caminho sem tools: adapter devolve SimpleNamespace(text=...)
    text, tool_calls, finish = pt.from_gemini_response(SimpleNamespace(text="olá mundo"))
    assert text == "olá mundo" and tool_calls is None and finish == "stop"


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------
def test_to_ollama_messages_converts_tool_call_args_to_object():
    msgs = pt.to_ollama_messages(MULTITURN)
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Natal"}
    assert msgs[2]["role"] == "tool" and msgs[2]["tool_name"] == "get_weather"


def test_from_ollama_chat_tool_calls():
    data = {
        "message": {
            "content": "",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Natal"}}}],
        }
    }
    text, tool_calls, finish = pt.from_ollama_chat(data)
    assert finish == "tool_calls" and text == ""
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Natal"}


def test_from_ollama_chat_plain_text():
    text, tool_calls, finish = pt.from_ollama_chat({"message": {"content": "resposta"}})
    assert text == "resposta" and tool_calls is None and finish == "stop"


# --------------------------------------------------------------------------
# Tools nativas (server-side) — pass-through por provedor
# --------------------------------------------------------------------------
def test_anthropic_native_server_tool_passes_through_next_to_function():
    out = pt.to_anthropic_tools(TOOLS + [ANTHROPIC_SERVER_TOOL])
    # A function tool continua traduzida (usa input_schema).
    assert out[0]["name"] == "get_weather" and "input_schema" in out[0]
    # O bloco server-side é repassado inalterado.
    assert ANTHROPIC_SERVER_TOOL in out
    assert len(out) == 2


def test_anthropic_native_only_tools_pass_through():
    out = pt.to_anthropic_tools([ANTHROPIC_SERVER_TOOL])
    assert out == [ANTHROPIC_SERVER_TOOL]


def test_anthropic_function_only_translation_unchanged():
    out = pt.to_anthropic_tools(TOOLS)
    assert len(out) == 1 and "input_schema" in out[0] and "type" not in out[0]


def test_gemini_merges_grounding_alongside_function_declarations():
    out = pt.to_gemini_tools(TOOLS + [GEMINI_GROUNDING_TOOL])
    # Grounding nativo mesclado inalterado.
    assert GEMINI_GROUNDING_TOOL in out
    # Bloco de function_declarations presente e traduzido.
    fdecls = [b for b in out if isinstance(b, dict) and "function_declarations" in b]
    assert fdecls and fdecls[0]["function_declarations"][0]["name"] == "get_weather"
    assert "additionalProperties" not in fdecls[0]["function_declarations"][0]["parameters"]


def test_gemini_native_only_grounding_pass_through():
    out = pt.to_gemini_tools([GEMINI_GROUNDING_TOOL, GEMINI_CODE_EXEC_TOOL])
    assert out == [GEMINI_GROUNDING_TOOL, GEMINI_CODE_EXEC_TOOL]
    # Sem function tools, não há bloco de declarações.
    assert all("function_declarations" not in b for b in out)


def test_gemini_function_only_output_unchanged():
    out = pt.to_gemini_tools(TOOLS)
    assert out == [{"function_declarations": [
        {"name": "get_weather", "description": "clima atual",
         "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
    ]}]


def test_openai_openrouter_native_tools_pass_through_untouched():
    # O caminho OpenAI/OpenRouter repassa o array canônico verbatim; a classificação
    # garante que a hosted tool não é descartada nem confundida com uma function tool.
    fns, natives = nt.split_tools(TOOLS + [OPENAI_HOSTED_TOOL])
    assert fns == TOOLS
    assert natives == [OPENAI_HOSTED_TOOL]
