# Objective: OpenAI-compatible chat completions facade for drop-in clients.
"""Map OpenAI /v1/chat/completions to the internal query router."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..api.auth import AuthContext, require_api_auth
from ..schemas import QueryRequest
from ..services.tenant_context import bind_tenant_to_request

router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: Optional[Union[str, List[Dict[str, Any]]]] = ""
    # Campos de tool calling (formato OpenAI)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    tenant_id: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


def _message_text(content: Any) -> str:
    """Extrai o texto de um ``content`` que pode ser str ou lista de partes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _messages_to_query(payload: ChatCompletionRequest) -> QueryRequest:
    raw_messages = [m.model_dump(exclude_none=True) for m in payload.messages]
    # Só tratamos como multi-turn quando há histórico de assistant/tool: aí as
    # `messages` são a fonte da verdade (e o RAG é ignorado). Um 1º turno (apenas
    # system/user), mesmo com tools, colapsa em `query` para permitir RAG.
    is_multiturn = any(m.get("role") in ("assistant", "tool") for m in raw_messages)

    system_parts: List[str] = []
    user_parts: List[str] = []
    last_user_text = ""
    for m in raw_messages:
        role = m.get("role")
        text = _message_text(m.get("content")).strip()
        if role == "system" and text:
            system_parts.append(text)
        elif role == "user" and text:
            user_parts.append(text)
            last_user_text = text
        elif role == "assistant" and text:
            user_parts.append(f"[assistant]\n{text}")

    if not any(m.get("role") == "user" for m in raw_messages):
        raise HTTPException(status_code=400, detail="messages deve conter ao menos uma mensagem user.")

    # `query` é usado para logging/RAG do 1º turno; em follow-up multi-turn as
    # `messages` são a fonte da verdade. Garante um valor não-vazio (schema exige).
    query_text = last_user_text or "\n\n".join(user_parts) or "(tool follow-up)"

    return QueryRequest(  # type: ignore[call-arg]  # pydantic: campos opcionais têm default
        query=query_text,
        modality="text",
        system_prompt="\n\n".join(system_parts) if system_parts else None,
        temperature=payload.temperature if payload.temperature is not None else 0.5,
        max_tokens=payload.max_tokens or 512,
        tenant_id=payload.tenant_id,
        stream=payload.stream,
        tools=payload.tools,
        tool_choice=payload.tool_choice,
        messages=raw_messages if is_multiturn else None,
    )


def _to_openai_response(result: Dict[str, Any], requested_model: Optional[str]) -> Dict[str, Any]:
    answer = str(result.get("answer", "") or "")
    model_used = str(result.get("model") or requested_model or "iag-router")
    metadata = result.get("metadata", {}) or {}
    tool_calls = result.get("tool_calls")
    finish_reason = result.get("finish_reason") or ("tool_calls" if tool_calls else "stop")

    message: Dict[str, Any] = {"role": "assistant", "content": answer or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_used,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(metadata.get("completion_tokens", 0) or 0),
            "total_tokens": int(metadata.get("total_tokens", 0) or 0),
        },
        "iag_router": {
            "estimated_cost_usd": result.get("estimated_cost_usd"),
            "correlation_id": metadata.get("correlation_id"),
        },
    }


@router.post("/v1/chat/completions", tags=["OpenAICompat"])
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    auth: AuthContext = Depends(require_api_auth),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """OpenAI-compatible chat endpoint backed by the internal router."""
    from ..services.query_http import execute_query, execute_query_stream

    req = bind_tenant_to_request(_messages_to_query(payload), auth)
    if idempotency_key:
        request.state.idempotency_key = idempotency_key

    if payload.stream:
        return await execute_query_stream(req, request, auth)

    result = await execute_query(req, request, auth)
    if isinstance(result, JSONResponse):
        return result
    body = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    return _to_openai_response(body, payload.model)
