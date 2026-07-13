# Objective: Tests for tenant binding and OpenAI-compatible API.
"""Production governance tests: tenant binding and OpenAI adapter."""

from __future__ import annotations

import pytest
from app.api.auth import AuthContext
from app.schemas import QueryRequest
from app.services.tenant_context import bind_tenant_to_request, resolve_effective_tenant_id
from fastapi import HTTPException


def test_resolve_tenant_from_auth():
    req = QueryRequest(query="hello")
    auth = AuthContext(authenticated=True, tenant_id="school-1", user_id="u1")
    assert resolve_effective_tenant_id(req, auth) == "school-1"


def test_tenant_mismatch_rejected(monkeypatch):
    monkeypatch.setattr(
        "app.services.tenant_context.settings.get",
        lambda key, default=None: "1" if key == "ENFORCE_TENANT_BINDING" else default,
    )
    req = QueryRequest(query="hello", tenant_id="other")
    auth = AuthContext(authenticated=True, tenant_id="school-1")
    with pytest.raises(HTTPException) as exc:
        resolve_effective_tenant_id(req, auth)
    assert exc.value.status_code == 403


def test_bind_tenant_sets_request_field():
    req = QueryRequest(query="hello")
    auth = AuthContext(authenticated=True, tenant_id="t-99")
    bound = bind_tenant_to_request(req, auth)
    assert bound.tenant_id == "t-99"


def test_openai_messages_to_query():
    from app.api.openai_compat_routes import ChatCompletionRequest, ChatMessage, _messages_to_query

    payload = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Olá"),
        ]
    )
    req = _messages_to_query(payload)
    assert req.query == "Olá"
    assert req.system_prompt == "You are helpful."
    assert req.modality == "text"
