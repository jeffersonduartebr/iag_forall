# Objective: Security regression tests for authentication and authorization.
"""Tests for API auth, job ownership, and RBAC hardening."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest
from app.api.auth import AuthContext, resolve_auth, verify_job_access
from app.api.deps import require_admin_or_role
from app.config.secrets_redaction import redact_secrets
from app.roadmap_features import check_access
from fastapi import HTTPException


def _make_jwt(secret: str, claims: dict) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    signing_input = f"{header}.{payload}".encode()
    signature = base64.urlsafe_b64encode(
        hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}.{signature}"


def test_resolve_auth_disabled_not_required(monkeypatch):
    monkeypatch.setenv("REQUIRE_API_AUTH", "0")
    from app.settings_dynamic import _lru

    _lru.clear()
    ctx = resolve_auth(required=False)
    assert ctx.authenticated is False


def test_resolve_auth_api_key(monkeypatch):
    monkeypatch.setenv("API_KEYS", "test-key-123")
    monkeypatch.setenv("REQUIRE_API_AUTH", "1")
    from app.settings_dynamic import _lru

    _lru.clear()
    ctx = resolve_auth(x_api_key="test-key-123", required=True)
    assert ctx.authenticated is True
    assert ctx.method == "api_key"


def test_resolve_auth_invalid_key_raises(monkeypatch):
    monkeypatch.setenv("API_KEYS", "valid-key")
    monkeypatch.setenv("REQUIRE_API_AUTH", "1")
    from app.settings_dynamic import _lru

    _lru.clear()
    with pytest.raises(HTTPException) as exc:
        resolve_auth(x_api_key="wrong-key", required=True)
    assert exc.value.status_code == 401


def test_jwt_roles_used_over_spoofed_header(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "jwt-test-secret")
    monkeypatch.setenv("TRUST_HEADER_ROLES", "0")
    from app.settings_dynamic import _lru

    _lru.clear()
    token = _make_jwt(
        "jwt-test-secret",
        {"roles": ["governance_admin"], "exp": int(time.time()) + 3600},
    )
    auth = require_admin_or_role(
        admin_token=None,
        user_id=None,
        user_roles_header="admin",
        required_roles=["governance_admin"],
        authorization=f"Bearer {token}",
    )
    assert auth["authorized_by"] == "jwt"
    assert "governance_admin" in auth["roles"]


def test_spoofed_header_roles_denied_without_trust(monkeypatch):
    monkeypatch.setenv("TRUST_HEADER_ROLES", "0")
    try:
        from app.settings_dynamic import _lru
        _lru.clear()
    except Exception:
        pass
    decision = check_access(
        user_id=None,
        required_roles=["governance_admin"],
        header_roles=["governance_admin"],
    )
    assert decision.allowed is False


def test_verify_job_access_requires_owner(monkeypatch):
    monkeypatch.setenv("REQUIRE_API_AUTH", "1")
    from app.settings_dynamic import _lru

    _lru.clear()
    owner = AuthContext(authenticated=True, method="api_key", api_key_hint="abc")
    record = {"owner_hash": owner.owner_hash()}
    other = AuthContext(authenticated=True, method="api_key", api_key_hint="xyz")

    verify_job_access(record, owner)
    with pytest.raises(HTTPException) as exc:
        verify_job_access(record, other)
    assert exc.value.status_code == 403


def test_redact_secrets_masks_passwords():
    snapshot = {"REDIS_PASSWORD": "secret", "NSGA_W_QUALITY": "0.5", "DB_PASS": "x"}
    redacted = redact_secrets(snapshot)
    assert redacted["REDIS_PASSWORD"] == "***REDACTED***"
    assert redacted["DB_PASS"] == "***REDACTED***"
    assert redacted["NSGA_W_QUALITY"] == "0.5"


@pytest.mark.asyncio
async def test_rag_healthcheck_embed_uses_thread_pool(monkeypatch):
    from unittest.mock import AsyncMock

    from app import rag_healthcheck

    calls = []

    def fake_embed(text):
        calls.append(text)
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(rag_healthcheck, "embed_text", fake_embed)
    monkeypatch.setattr(rag_healthcheck, "embed_multimodal", lambda *a, **k: [0.1])
    monkeypatch.setattr(
        rag_healthcheck,
        "get_or_create_collection_async",
        AsyncMock(return_value=object()),
    )
    monkeypatch.setattr(rag_healthcheck, "add_document", AsyncMock(return_value=True))
    monkeypatch.setattr(
        rag_healthcheck,
        "query_embedding",
        AsyncMock(return_value={"documents": [["ok"]]}),
    )
    monkeypatch.setattr(rag_healthcheck, "get_redis", lambda: None)

    report = await rag_healthcheck.rag_healthcheck()
    assert report["steps"]["embeddings_text"]["ok"] is True
    assert len(calls) == 2
