# Objective: Integration-ish tests for tool/function calling across schemas, routing and providers.
"""Testes de tool/function calling: contratos, capacidade e providers (mockados)."""

from types import SimpleNamespace

import pybreaker
import pytest
from app.schemas import QueryRequest, QueryResponse

from app import providers_async as pa

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "clima atual",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]


def _close_breakers():
    pa.cloud_breaker._state = pybreaker.CircuitClosedState(pa.cloud_breaker)
    pa.local_breaker._state = pybreaker.CircuitClosedState(pa.local_breaker)


# --------------------------------------------------------------------------
# Contratos (schemas)
# --------------------------------------------------------------------------
def test_query_request_accepts_tool_fields():
    req = QueryRequest(query="clima?", tools=TOOLS, tool_choice="auto", messages=[{"role": "user", "content": "clima?"}])
    assert req.tools == TOOLS
    assert req.tool_choice == "auto"
    assert req.messages[0]["role"] == "user"


def test_query_request_tools_default_none():
    req = QueryRequest(query="oi")
    assert req.tools is None and req.tool_choice is None and req.messages is None


def test_query_response_round_trips_tool_calls():
    tcs = [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]
    resp = QueryResponse(answer="", model="ollama/qwen2.5", tool_calls=tcs, finish_reason="tool_calls")
    dumped = resp.model_dump()
    assert dumped["tool_calls"] == tcs
    assert dumped["finish_reason"] == "tool_calls"


# --------------------------------------------------------------------------
# Roteamento por capacidade
# --------------------------------------------------------------------------
def test_model_config_supports_tools_property():
    from app.model_registry import Capability, ModelConfig, Provider

    with_tools = ModelConfig(name="m", provider=Provider.OPENAI, capabilities={Capability.TEXT, Capability.FUNCTION_CALLING})
    without = ModelConfig(name="m2", provider=Provider.OLLAMA, capabilities={Capability.TEXT})
    assert with_tools.supports_tools is True
    assert without.supports_tools is False


def test_model_supports_tools_uses_registry():
    from app.model_registry import model_supports_tools

    # gpt-4o tem FUNCTION_CALLING nos modelos default do registry
    assert model_supports_tools("openai/gpt-4o") is True
    # modelo desconhecido/local sem sinal explícito → False (estrito)
    assert model_supports_tools("ollama/some-unknown-model") is False


def test_filter_tool_capable_model_names():
    from app.model_registry import filter_tool_capable_model_names

    result = filter_tool_capable_model_names(["openai/gpt-4o", "ollama/unknown-model"])
    assert result == ["openai/gpt-4o"]


# --------------------------------------------------------------------------
# Fachada OpenAI-compat
# --------------------------------------------------------------------------
def test_openai_compat_maps_tools_and_multiturn():
    from app.api.openai_compat_routes import ChatCompletionRequest, _messages_to_query

    payload = ChatCompletionRequest(
        model="ollama/qwen2.5",
        messages=[
            {"role": "user", "content": "clima em Natal?"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Natal"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "32C"},
        ],
        tools=TOOLS,
    )
    req = _messages_to_query(payload)
    assert req.tools == TOOLS
    # follow-up multi-turn → messages preservado como fonte da verdade
    assert req.messages is not None and len(req.messages) == 3
    # query derivada da última mensagem user (para logging), não vazia
    assert req.query.strip() != ""


def test_openai_compat_first_turn_collapses_without_history():
    from app.api.openai_compat_routes import ChatCompletionRequest, _messages_to_query

    payload = ChatCompletionRequest(messages=[{"role": "user", "content": "clima?"}], tools=TOOLS)
    req = _messages_to_query(payload)
    # 1º turno com tools mas sem histórico assistant/tool → messages None (permite RAG)
    assert req.messages is None
    assert req.tools == TOOLS
    assert req.query == "clima?"


def test_openai_compat_response_emits_tool_calls():
    from app.api.openai_compat_routes import _to_openai_response

    tcs = [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]
    body = _to_openai_response({"answer": "", "model": "gpt-4o", "tool_calls": tcs, "finish_reason": "tool_calls"}, "gpt-4o")
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == tcs
    assert choice["message"]["content"] is None


# --------------------------------------------------------------------------
# Provider OpenAI (mockado): tools entram, tool_calls saem
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_openai_provider_forwards_tools_and_returns_tool_calls(monkeypatch):
    _close_breakers()
    captured = {}

    async def _create(**kwargs):
        captured.update(kwargs)
        msg = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(id="c1", type="function", function=SimpleNamespace(name="get_weather", arguments='{"city":"Natal"}'))],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
            model_dump=lambda: {},
        )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))
    monkeypatch.setattr(pa, "AsyncOpenAI", lambda api_key=None: fake_client)
    monkeypatch.setattr(pa, "get_model_cost", lambda m, p, c: 0.1)

    provider = pa.OpenAIProvider()
    out = await provider.generate(prompt="clima em Natal?", model="gpt-4o", tools=TOOLS, tool_choice="auto")

    # tools repassadas ao SDK
    assert captured["tools"] == TOOLS and captured["tool_choice"] == "auto"
    # tool_calls normalizados na saída
    assert out.finish_reason == "tool_calls"
    assert out.tool_calls[0]["function"]["name"] == "get_weather"
    assert out.text == ""

    # _build_response_meta propaga tool_calls/finish_reason
    meta = pa._build_response_meta(out)
    assert meta["tool_calls"] == out.tool_calls
    assert meta["finish_reason"] == "tool_calls"


# --------------------------------------------------------------------------
# Provider Ollama (mockado): tools → /api/chat
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ollama_provider_uses_chat_endpoint_for_tools(monkeypatch):
    _close_breakers()
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {"content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Natal"}}}]},
                "prompt_eval_count": 5,
                "eval_count": 0,
                "load_duration": 0,
            }

    class _Client:
        async def post(self, url, json=None, timeout=None):
            captured["url"] = url
            captured["payload"] = json
            return _Resp()

    async def _get_client():
        return _Client()

    monkeypatch.setattr(pa, "get_http_client", _get_client)
    monkeypatch.setattr(pa, "get_model_cost", lambda m, p, c: 0.0)

    provider = pa.OllamaProvider()
    out = await provider.generate(prompt="clima em Natal?", model="qwen2.5", tools=TOOLS, tool_choice="auto")

    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["tools"] == TOOLS
    assert out.finish_reason == "tool_calls"
    assert out.tool_calls[0]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_ollama_provider_uses_generate_endpoint_without_tools(monkeypatch):
    _close_breakers()
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "4", "prompt_eval_count": 3, "eval_count": 1, "load_duration": 0}

    class _Client:
        async def post(self, url, json=None, timeout=None):
            captured["url"] = url
            return _Resp()

    async def _get_client():
        return _Client()

    monkeypatch.setattr(pa, "get_http_client", _get_client)
    monkeypatch.setattr(pa, "get_model_cost", lambda m, p, c: 0.0)

    provider = pa.OllamaProvider()
    out = await provider.generate(prompt="2+2?", model="gemma3:4b")

    assert captured["url"].endswith("/api/generate")
    assert out.text == "4" and out.tool_calls is None


# --------------------------------------------------------------------------
# End-to-end: process_query_request tramita tools e expõe tool_calls
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_process_query_request_threads_tools_and_surfaces_tool_calls(monkeypatch):
    from app.services import query_runtime as qr

    captured = {}

    async def _allow_guardrails_async(_query):
        return SimpleNamespace(allowed=True, reasons=[])

    async def _allow_budget_async(_tenant=None):
        return SimpleNamespace(
            allowed=True, reason=None, daily_spent=0, monthly_spent=0, daily_limit=0, monthly_limit=0
        )

    async def _no_policy():
        return None

    async def _route(**kwargs):
        captured.update(kwargs)
        return {
            "answer": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Natal"}'}}],
            "finish_reason": "tool_calls",
            "model": "openai/gpt-4o",
            "modality": "text",
            "latency_s": 0.1,
            "cost_per_1k": 0.0,
            "estimated_cost_usd": 0.0,
            "metadata": {"prompt_tokens": 5, "completion_tokens": 2, "uncertainty_score": 0.3},
            "route": {"fallback": {"used": False}},
            "candidates": [],
        }

    monkeypatch.setattr(qr, "check_input_guardrails_async", _allow_guardrails_async)
    monkeypatch.setattr(qr, "check_tenant_budget", _allow_budget_async)
    monkeypatch.setattr(qr, "get_active_policy", _no_policy)
    monkeypatch.setattr(qr, "sanitize_output_guardrails", lambda answer: (answer, []))
    monkeypatch.setattr(qr, "route_and_answer", _route)

    req = QueryRequest(query="clima em Natal?", tools=TOOLS, tool_choice="auto")
    out = await qr.process_query_request(req)
    result = out["result"]

    # tools chegaram ao roteador
    assert captured["tools"] == TOOLS and captured["tool_choice"] == "auto"
    # tool_calls/finish_reason expostos no resultado
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    # guards: sem abstenção/substituição por texto de "sem evidência"
    assert result["abstained"] is False
    assert result["answer"] == ""
    assert result["review_status"] == "auto_approved"
