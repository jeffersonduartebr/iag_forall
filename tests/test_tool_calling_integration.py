# Objective: Automated lane for tool/function calling round-trips (facade + live).
"""Testes de integração de tool/function calling (roadmap #12).

Cobre três frentes:

1. Round-trip end-to-end na fachada OpenAI-compat (``/v1/chat/completions``)
   com provider mockado via ``route_and_answer``: 1º turno devolve ``tool_calls``
   e o follow-up (com resultado da tool) devolve texto final.
2. Contrato da resposta OpenAI Chat Completions (``_to_openai_response``).
3. Teste vivo (``@pytest.mark.integration``, fora da lane padrão) contra um
   Ollama local com modelo capaz de tools; pula limpo se o Ollama não responder.
"""

from __future__ import annotations

import json
import os

import pytest
from app.api import openai_compat_routes as oc
from app.api.auth import AuthContext
from app.services import query_runtime as qr

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Consulta o clima atual de uma cidade.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

TOOL_CALL = {
    "id": "call_natal_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Natal"}'},
}


def _tool_turn_result() -> dict:
    """Router result shaped like a tool-call turn (empty text + tool_calls)."""
    return {
        "answer": "",
        "tool_calls": [TOOL_CALL],
        "finish_reason": "tool_calls",
        "model": "ollama/qwen2.5",
        "modality": "text",
        "latency_s": 0.05,
        "cost_per_1k": 0.0,
        "estimated_cost_usd": 0.0,
        "metadata": {"prompt_tokens": 12, "completion_tokens": 8, "uncertainty_score": 0.3},
        "route": {"chosen_model": "ollama/qwen2.5", "fallback": {"used": False}},
        "candidates": [],
    }


def _text_turn_result(text: str) -> dict:
    """Router result shaped like a final text answer (post tool execution)."""
    return {
        "answer": text,
        "tool_calls": None,
        "finish_reason": "stop",
        "model": "ollama/qwen2.5",
        "modality": "text",
        "latency_s": 0.05,
        "cost_per_1k": 0.0,
        "estimated_cost_usd": 0.0,
        "metadata": {"prompt_tokens": 20, "completion_tokens": 9, "uncertainty_score": 0.2},
        "route": {"chosen_model": "ollama/qwen2.5", "fallback": {"used": False}},
        "candidates": [],
    }


def _patch_query_runtime(monkeypatch, route_fn) -> None:
    """Neutralize guardrails/budget/policy and swap the router entrypoint.

    Espelha ``test_tool_calling.test_process_query_request_threads_tools_and_
    surfaces_tool_calls``: só o ``route_and_answer`` muda de comportamento; o
    restante do pipeline (perfil de workload, enrich, side-effects) roda de fato.
    """
    from types import SimpleNamespace

    async def _allow_guardrails_async(_query):
        return SimpleNamespace(allowed=True, reasons=[])

    async def _allow_budget_async(_tenant=None):
        return SimpleNamespace(
            allowed=True, reason=None, daily_spent=0, monthly_spent=0, daily_limit=0, monthly_limit=0
        )

    async def _no_policy():
        return None

    monkeypatch.setattr(qr, "check_input_guardrails_async", _allow_guardrails_async)
    monkeypatch.setattr(qr, "check_tenant_budget", _allow_budget_async)
    monkeypatch.setattr(qr, "get_active_policy", _no_policy)
    monkeypatch.setattr(qr, "sanitize_output_guardrails", lambda answer: (answer, []))
    monkeypatch.setattr(qr, "route_and_answer", route_fn)


# --------------------------------------------------------------------------
# 1. Round-trip end-to-end na fachada OpenAI-compat (unit lane, mockado)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_openai_compat_facade_tool_call_round_trip(monkeypatch):
    """1º turno → tool_calls; follow-up com role:tool → texto final."""
    calls = {"n": 0}

    async def _route(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # tools precisam chegar ao roteador no 1º turno
            assert kwargs["tools"] == TOOLS
            assert kwargs["tool_choice"] == "auto"
            return _tool_turn_result()
        # follow-up multi-turn: messages são a fonte da verdade
        assert kwargs["messages"] is not None
        assert any(m.get("role") == "tool" for m in kwargs["messages"])
        return _text_turn_result("Em Natal está 32C e ensolarado.")

    _patch_query_runtime(monkeypatch, _route)
    auth = AuthContext()

    # --- Turno 1: usuário pergunta, modelo pede a tool ---
    first = oc.ChatCompletionRequest(
        model="ollama/qwen2.5",
        messages=[{"role": "user", "content": "Qual o clima em Natal?"}],
        tools=TOOLS,
        tool_choice="auto",
    )
    body1 = await oc.chat_completions(first, request=None, auth=auth, idempotency_key=None)

    assert body1["object"] == "chat.completion"
    choice1 = body1["choices"][0]
    assert choice1["finish_reason"] == "tool_calls"
    assert choice1["message"]["content"] is None
    tool_calls = choice1["message"]["tool_calls"]
    assert tool_calls and tool_calls[0]["function"]["name"] == "get_weather"

    returned = tool_calls[0]
    # --- Turno 2: cliente devolve o resultado da tool, espera texto final ---
    follow_up = oc.ChatCompletionRequest(
        model="ollama/qwen2.5",
        messages=[
            {"role": "user", "content": "Qual o clima em Natal?"},
            {"role": "assistant", "content": "", "tool_calls": [returned]},
            {
                "role": "tool",
                "tool_call_id": returned["id"],
                "name": returned["function"]["name"],
                "content": "32C, ensolarado",
            },
        ],
        tools=TOOLS,
    )
    body2 = await oc.chat_completions(follow_up, request=None, auth=auth, idempotency_key=None)

    choice2 = body2["choices"][0]
    assert choice2["finish_reason"] == "stop"
    assert choice2["message"]["content"] == "Em Natal está 32C e ensolarado."
    assert choice2["message"].get("tool_calls") is None
    assert calls["n"] == 2


# --------------------------------------------------------------------------
# 2. Contrato da resposta OpenAI Chat Completions
# --------------------------------------------------------------------------
def test_to_openai_response_matches_chat_completion_contract():
    """A resposta emitida bate com o contrato público OpenAI (tool call turn)."""
    result = {
        "answer": "",
        "model": "ollama/qwen2.5",
        "tool_calls": [TOOL_CALL],
        "finish_reason": "tool_calls",
        "estimated_cost_usd": 0.0,
        "metadata": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "correlation_id": "cid-abc",
        },
    }

    body = oc._to_openai_response(result, "ollama/qwen2.5")

    # Envelope
    assert isinstance(body["id"], str) and body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert isinstance(body["created"], int)
    assert body["model"] == "ollama/qwen2.5"

    # Choice + message
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["role"] == "assistant"
    assert message["content"] is None

    # tool_calls no formato OpenAI: id/type/function.name/function.arguments(str JSON)
    tc = message["tool_calls"][0]
    assert tc["id"] == "call_natal_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"city": "Natal"}

    # usage
    usage = body["usage"]
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == 18


def test_to_openai_response_defaults_finish_reason_for_plain_text():
    """Sem tool_calls e sem finish_reason explícito → 'stop' com content textual."""
    body = oc._to_openai_response({"answer": "olá", "model": "ollama/qwen2.5", "metadata": {}}, None)
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "olá"
    assert "tool_calls" not in choice["message"]


# --------------------------------------------------------------------------
# 3. Round-trip vivo contra Ollama local (fora da lane padrão)
# --------------------------------------------------------------------------
def _ollama_tags(host: str) -> list[str]:
    """Return installed model names or raise if Ollama is unreachable."""
    import httpx

    resp = httpx.get(f"{host}/api/tags", timeout=3.0)
    resp.raise_for_status()
    return [m.get("name", "") for m in resp.json().get("models", [])]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_ollama_tool_round_trip():
    """Drive a real tool round-trip against a local tool-capable Ollama model."""
    from app import providers_async as pa

    host = pa.OLLAMA_HOST
    model = os.getenv("OLLAMA_TOOL_MODEL", "ollama/qwen2.5")
    bare = model.split("/", 1)[1] if "/" in model else model

    try:
        tags = _ollama_tags(host)
    except Exception as exc:  # ConnectionError, timeout, HTTP error, etc.
        pytest.skip(f"Ollama unreachable at {host}: {exc}")

    base = bare.split(":")[0]
    if not any(base in (tag or "") for tag in tags):
        pytest.skip(f"model '{model}' not pulled in Ollama (available: {tags})")

    tools = TOOLS
    question = "What's the weather in Natal, Brazil? Use the get_weather tool."

    # --- Turno 1: modelo deve emitir tool_calls ---
    _text, meta = await pa.call_model(
        model, question, tools=tools, tool_choice="auto", max_tokens=256, timeout_seconds=60
    )
    assert meta["finish_reason"] == "tool_calls", f"esperava tool_calls, veio {meta}"
    assert meta["tool_calls"], "provider não retornou tool_calls"
    call = meta["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    call_id = call.get("id") or "call_1"
    fname = call["function"]["name"]

    # --- Turno 2: devolvemos o resultado da tool e esperamos texto ---
    messages = [
        {"role": "user", "content": "What's the weather in Natal, Brazil?"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call_id, "name": fname, "content": "32C, sunny"},
    ]
    text2, meta2 = await pa.call_model(
        model,
        "What's the weather in Natal, Brazil?",
        tools=tools,
        messages=messages,
        max_tokens=256,
        timeout_seconds=60,
    )
    assert (text2 or "").strip() != "", f"esperava texto final, veio {meta2}"
    assert meta2.get("finish_reason") in (None, "stop")
