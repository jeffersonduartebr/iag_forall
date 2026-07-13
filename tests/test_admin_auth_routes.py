# Objective: Test coverage for admin UI authentication routes.
"""Tests for admin web console login and session endpoints."""

import time

import pytest
from app.api.admin_auth_routes import (
    _LOGIN_ATTEMPTS,
    LoginRequest,
    admin_login,
    admin_me,
    expert_login,
    expert_me,
)
from app.schemas import ExpertLoginRequest
from fastapi import HTTPException


def _patch_settings(monkeypatch, mapping):
    from app.settings_dynamic import _lru, settings

    _lru.clear()
    stored = dict(mapping)
    original_get = settings.get

    def patched_get(key, fallback=None):
        if key in stored:
            return stored[key]
        return original_get(key, fallback)

    monkeypatch.setattr(settings, "get", patched_get)
    for key, value in mapping.items():
        monkeypatch.setenv(key, str(value))


@pytest.fixture(autouse=True)
def _clear_login_attempts(monkeypatch):
    from app.settings_dynamic import _lru

    _lru.clear()
    _LOGIN_ATTEMPTS.clear()
    monkeypatch.delenv("ADMIN_UI_PASSWORD", raising=False)
    monkeypatch.delenv("ADMIN_UI_USERNAME", raising=False)
    yield
    _LOGIN_ATTEMPTS.clear()
    _lru.clear()


def test_admin_login_success(monkeypatch):
    """Valid credentials should return a bearer JWT."""
    _patch_settings(
        monkeypatch,
        {
            "ADMIN_UI_USERNAME": "jefferson.silva",
            "ADMIN_UI_PASSWORD": "abc@123",
            "JWT_SECRET": "test-secret",
            "ADMIN_UI_SESSION_TTL_S": 3600,
        },
    )
    out = admin_login(LoginRequest(username="jefferson.silva", password="abc@123"), x_forwarded_for=None, x_real_ip=None)
    assert out["token_type"] == "bearer"
    assert out["access_token"]
    assert out["username"] == "jefferson.silva"
    assert out["expires_at"] > int(time.time())


def test_admin_login_invalid_credentials(monkeypatch):
    """Wrong password should return 401."""
    _patch_settings(
        monkeypatch,
        {
            "ADMIN_UI_USERNAME": "jefferson.silva",
            "ADMIN_UI_PASSWORD": "abc@123",
            "JWT_SECRET": "test-secret",
        },
    )
    with pytest.raises(HTTPException) as exc:
        admin_login(LoginRequest(username="jefferson.silva", password="wrong"), x_forwarded_for=None, x_real_ip=None)
    assert exc.value.status_code == 401


def test_admin_login_rate_limit(monkeypatch):
    """Repeated failures should trigger 429."""
    _patch_settings(
        monkeypatch,
        {
            "ADMIN_UI_USERNAME": "jefferson.silva",
            "ADMIN_UI_PASSWORD": "abc@123",
            "JWT_SECRET": "test-secret",
        },
    )
    for _ in range(10):
        with pytest.raises(HTTPException):
            admin_login(LoginRequest(username="jefferson.silva", password="bad"), x_forwarded_for="1.2.3.4", x_real_ip=None)
    with pytest.raises(HTTPException) as exc:
        admin_login(LoginRequest(username="jefferson.silva", password="bad"), x_forwarded_for="1.2.3.4", x_real_ip=None)
    assert exc.value.status_code == 429


def test_admin_me_with_jwt(monkeypatch):
    """JWT from login should authorize /admin/auth/me."""
    _patch_settings(
        monkeypatch,
        {
            "ADMIN_UI_USERNAME": "jefferson.silva",
            "ADMIN_UI_PASSWORD": "abc@123",
            "JWT_SECRET": "test-secret",
        },
    )
    login = admin_login(LoginRequest(username="jefferson.silva", password="abc@123"), x_forwarded_for=None, x_real_ip=None)
    me = admin_me(authorization=f"Bearer {login['access_token']}")
    assert me["username"] == "jefferson.silva"
    assert "admin" in me["roles"]


def test_expert_login_success(monkeypatch):
    """Expert credentials should return a limited-role JWT."""
    _patch_settings(
        monkeypatch,
        {
            "EXPERT_UI_USERS": '{"maria.historia":"secret123"}',
            "JWT_SECRET": "test-secret",
            "EXPERT_UI_SESSION_TTL_S": 3600,
        },
    )
    # Sem conta no DB: cai no caminho legado (EXPERT_UI_USERS) sem tocar o MySQL.
    monkeypatch.setattr("app.services.expert_accounts.authenticate_expert_account", lambda e, p: None)
    out = expert_login(
        ExpertLoginRequest(email="maria.historia", password="secret123"),
        x_forwarded_for=None,
        x_real_ip=None,
    )
    assert out["portal"] == "expert"
    assert out["access_token"]
    me = expert_me(authorization=f"Bearer {out['access_token']}")
    assert me["username"] == "maria.historia"
    assert me["roles"] == ["expert_reviewer"]
    assert me["portal"] == "expert"


def test_expert_login_db_account(monkeypatch):
    """DB-backed expert accounts should authenticate by email."""
    _patch_settings(monkeypatch, {"JWT_SECRET": "test-secret"})

    def fake_auth(email, password):
        if email == "maria@uni.edu" and password == "senha1234":
            return {"email": "maria@uni.edu", "display_name": "Maria", "enabled": 1}
        return None

    monkeypatch.setattr("app.services.expert_accounts.authenticate_expert_account", fake_auth)
    out = expert_login(
        ExpertLoginRequest(email="maria@uni.edu", password="senha1234"),
        x_forwarded_for=None,
        x_real_ip=None,
    )
    assert out["email"] == "maria@uni.edu"
    assert out["access_token"]


def test_authenticate_expert_login_db_first(monkeypatch):
    # Importa admin_auth_routes/expert_accounts do sys.modules atual e usa por eles:
    # outros arquivos fazem sys.modules.pop(...) na coleção, o que pode deixar o
    # `settings`/`authenticate` de um objeto de módulo antigo diferente do que
    # `_patch_settings` (que importa o settings atual) alcança.
    from app.api import admin_auth_routes as aar
    from app.services import expert_accounts as ea

    _patch_settings(monkeypatch, {"EXPERT_UI_USERS": '{"legacy":"pass"}'})

    def fake_auth(email, password):
        if email == "a@b.com":
            return {"email": "a@b.com"}
        return None

    monkeypatch.setattr(ea, "authenticate_expert_account", fake_auth)
    assert aar._authenticate_expert_login("a@b.com", "x") == "a@b.com"
    assert aar._authenticate_expert_login("legacy", "pass") == "legacy"
    assert aar._authenticate_expert_login("missing", "x") is None


def test_expert_login_not_configured(monkeypatch):
    """Missing credentials should return 401."""
    _patch_settings(monkeypatch, {"JWT_SECRET": "test-secret"})
    monkeypatch.setattr("app.api.admin_auth_routes._parse_expert_users", lambda: {})
    monkeypatch.setattr("app.services.expert_accounts.authenticate_expert_account", lambda e, p: None)
    with pytest.raises(HTTPException) as exc:
        expert_login(ExpertLoginRequest(email="x@y.com", password="y"), x_forwarded_for=None, x_real_ip=None)
    assert exc.value.status_code == 401


def test_expert_login_invalid_credentials(monkeypatch):
    _patch_settings(
        monkeypatch,
        {
            "EXPERT_UI_USERS": '{"maria.historia":"secret123"}',
            "JWT_SECRET": "test-secret",
        },
    )
    monkeypatch.setattr("app.services.expert_accounts.authenticate_expert_account", lambda e, p: None)
    with pytest.raises(HTTPException) as exc:
        expert_login(
            ExpertLoginRequest(email="maria.historia", password="wrong"),
            x_forwarded_for=None,
            x_real_ip=None,
        )
    assert exc.value.status_code == 401
