# Objective: Admin UI session authentication (login/password).
"""Login endpoints for the admin web console."""

from __future__ import annotations

import time
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from ..schemas import ExpertLoginRequest
from ..settings_dynamic import settings
from ..utils.client_ip import parse_trusted_proxies, resolve_client_ip
from ..utils.secure_compare import secret_equals
from .auth import _auth_from_jwt, _encode_jwt_hs256, _extract_bearer_token
from .dependencies import admin_session

router = APIRouter()

_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
_MAX_ATTEMPTS = 10
_WINDOW_S = 300
_MAX_TRACKED_CLIENTS = 10_000  # teto do dicionário de tentativas (evicção FIFO)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


def _client_key(request: Optional[Request], x_forwarded_for: Optional[str], x_real_ip: Optional[str]) -> str:
    """Throttling key: the peer IP, or the forwarded client only behind a trusted proxy."""
    forwarded = x_forwarded_for or x_real_ip
    direct_ip = request.client.host if request is not None and request.client else None
    return resolve_client_ip(direct_ip, forwarded, parse_trusted_proxies(settings.get("TRUSTED_PROXY_IPS", "")))


def _check_rate_limit(key: str) -> None:
    now = time.time()
    attempts = [t for t in _LOGIN_ATTEMPTS.get(key, []) if now - t < _WINDOW_S]
    if attempts:
        _LOGIN_ATTEMPTS[key] = attempts
    else:
        _LOGIN_ATTEMPTS.pop(key, None)  # não guarda chaves sem tentativas recentes
    if len(attempts) >= _MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Muitas tentativas. Tente novamente em alguns minutos.")


def _record_failed_login(key: str) -> None:
    _LOGIN_ATTEMPTS.setdefault(key, []).append(time.time())
    while len(_LOGIN_ATTEMPTS) > _MAX_TRACKED_CLIENTS:
        _LOGIN_ATTEMPTS.pop(next(iter(_LOGIN_ATTEMPTS)))


def _jwt_secret() -> str:
    secret = (settings.get("JWT_SECRET", "") or "").strip()
    if secret:
        return secret
    return (settings.ADMIN_TOKEN or "").strip()


def _validate_ui_credentials(username: str, password: str) -> bool:
    expected_user = (settings.get("ADMIN_UI_USERNAME", "admin") or "admin").strip()
    expected_pass = (settings.get("ADMIN_UI_PASSWORD", "") or "").strip()
    if not expected_pass:
        return False
    user_ok = secret_equals(username.strip(), expected_user)
    pass_ok = secret_equals(password, expected_pass)
    return user_ok and pass_ok


def _parse_expert_users() -> dict[str, str]:
    """Load expert portal accounts from EXPERT_UI_USERS JSON or single-user env."""
    import json

    raw = (settings.get("EXPERT_UI_USERS", "") or "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return {str(k).strip(): str(v) for k, v in data.items() if str(k).strip() and str(v)}
        except Exception:
            pass
    user = (settings.get("EXPERT_UI_USERNAME", "") or "").strip()
    passwd = (settings.get("EXPERT_UI_PASSWORD", "") or "").strip()
    if user and passwd:
        return {user: passwd}
    return {}


def _validate_expert_credentials(username: str, password: str) -> bool:
    users = _parse_expert_users()
    expected = users.get(username.strip())
    if not expected:
        return False
    return secret_equals(password, expected)


def _issue_admin_token(username: str) -> tuple[str, int]:
    secret = _jwt_secret()
    if not secret:
        raise HTTPException(status_code=503, detail="JWT_SECRET ou ADMIN_TOKEN necessário para sessão admin.")
    ttl = int(settings.get("ADMIN_UI_SESSION_TTL_S", 28800) or 28800)
    now = int(time.time())
    exp = now + max(300, ttl)
    payload = {
        "sub": username,
        "roles": ["admin", "platform_admin", "governance_admin", "governance_viewer", "policy_admin", "audit_viewer", "expert_reviewer", "eval_viewer", "eval_admin", "researcher"],
        "iat": now,
        "exp": exp,
    }
    return _encode_jwt_hs256(payload, secret), exp


def _issue_expert_token(username: str) -> tuple[str, int]:
    secret = _jwt_secret()
    if not secret:
        raise HTTPException(status_code=503, detail="JWT_SECRET ou ADMIN_TOKEN necessário para sessão expert.")
    ttl = int(settings.get("EXPERT_UI_SESSION_TTL_S", settings.get("ADMIN_UI_SESSION_TTL_S", 28800)) or 28800)
    now = int(time.time())
    exp = now + max(300, ttl)
    payload = {
        "sub": username,
        "roles": ["expert_reviewer"],
        "portal": "expert",
        "iat": now,
        "exp": exp,
    }
    return _encode_jwt_hs256(payload, secret), exp


def resolve_admin_session(
    *,
    x_admin_token: Optional[str] = None,
    authorization: Optional[str] = None,
) -> dict:
    """Accept X-Admin-Token or admin JWT bearer."""
    configured = (settings.ADMIN_TOKEN or "").strip()
    previous = (settings.ADMIN_TOKEN_PREVIOUS or "").strip()
    if configured and x_admin_token:
        if secret_equals(x_admin_token, configured) or (previous and secret_equals(x_admin_token, previous)):
            return {"authorized_by": "admin_token", "username": "admin", "roles": ["admin"]}

    token = _extract_bearer_token(authorization)
    if token:
        ctx = _auth_from_jwt(token)
        if ctx and ctx.authenticated and ("admin" in ctx.roles or "platform_admin" in ctx.roles):
            return {
                "authorized_by": "admin_jwt",
                "username": ctx.user_id or "admin",
                "roles": list(ctx.roles),
            }
        if configured and secret_equals(token, configured):
            return {"authorized_by": "admin_token", "username": "admin", "roles": ["admin"]}

    raise HTTPException(status_code=401, detail="Não autorizado.")


def resolve_expert_session(
    *,
    x_admin_token: Optional[str] = None,
    authorization: Optional[str] = None,
) -> dict:
    """Accept expert JWT or full admin credentials for the expert portal."""
    configured = (settings.ADMIN_TOKEN or "").strip()
    previous = (settings.ADMIN_TOKEN_PREVIOUS or "").strip()
    if configured and x_admin_token:
        if secret_equals(x_admin_token, configured) or (previous and secret_equals(x_admin_token, previous)):
            return {"authorized_by": "admin_token", "username": "admin", "roles": ["admin", "expert_reviewer"], "portal": "admin"}

    token = _extract_bearer_token(authorization)
    if token:
        ctx = _auth_from_jwt(token)
        if ctx and ctx.authenticated:
            roles = list(ctx.roles)
            if "expert_reviewer" in roles:
                return {
                    "authorized_by": "expert_jwt",
                    "username": ctx.user_id or "expert",
                    "roles": roles,
                    "portal": "expert",
                }
            if "admin" in roles or "platform_admin" in roles:
                return {
                    "authorized_by": "admin_jwt",
                    "username": ctx.user_id or "admin",
                    "roles": roles,
                    "portal": "admin",
                }
        if configured and secret_equals(token, configured):
            return {"authorized_by": "admin_token", "username": "admin", "roles": ["admin"], "portal": "admin"}

    raise HTTPException(status_code=401, detail="Não autorizado.")


@router.post("/admin/auth/login", tags=["AdminAuth"])
def admin_login(
    payload: LoginRequest,
    x_forwarded_for: Annotated[Optional[str], Header()] = None,
    x_real_ip: Annotated[Optional[str], Header()] = None,
    request: Request = None,  # type: ignore[assignment]  # injetado pelo FastAPI
):
    """Authenticate admin UI user and return JWT."""
    key = _client_key(request, x_forwarded_for, x_real_ip)
    _check_rate_limit(key)
    if not _validate_ui_credentials(payload.username, payload.password):
        _record_failed_login(key)
        raise HTTPException(status_code=401, detail="Credenciais inválidas.")
    access_token, exp = _issue_admin_token(payload.username.strip())
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_at": exp,
        "username": payload.username.strip(),
        "portal": "admin",
    }


def _authenticate_expert_login(email_or_username: str, password: str) -> Optional[str]:
    """Try DB account auth first, then legacy env-based accounts."""
    from ..services.expert_accounts import authenticate_expert_account, normalize_email

    normalized = normalize_email(email_or_username)
    account = authenticate_expert_account(normalized, password)
    if account:
        return str(account.get("email") or normalized)

    # Legacy env login accepts username keys (not necessarily email)
    users = _parse_expert_users()
    expected = users.get(email_or_username.strip())
    if expected and secret_equals(password, expected):
        return email_or_username.strip()
    return None


@router.post("/admin/auth/expert-login", tags=["AdminAuth"])
def expert_login(
    payload: ExpertLoginRequest,
    x_forwarded_for: Annotated[Optional[str], Header()] = None,
    x_real_ip: Annotated[Optional[str], Header()] = None,
    request: Request = None,  # type: ignore[assignment]  # injetado pelo FastAPI
):
    """Authenticate expert portal user with limited reviewer role."""
    key = f"expert:{_client_key(request, x_forwarded_for, x_real_ip)}"
    _check_rate_limit(key)

    identity = _authenticate_expert_login(payload.email, payload.password)
    if not identity:
        _record_failed_login(key)
        raise HTTPException(status_code=401, detail="Credenciais inválidas.")

    access_token, exp = _issue_expert_token(identity)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_at": exp,
        "username": identity,
        "email": identity,
        "portal": "expert",
    }


@router.post("/admin/auth/logout", tags=["AdminAuth"])
def admin_logout():
    """Logout is client-side (drop JWT)."""
    return {"status": "ok"}


@router.get("/admin/auth/me", tags=["AdminAuth"])
def admin_me(session: dict = Depends(admin_session)):
    """Return current admin session identity."""
    return {
        "username": session.get("username", "admin"),
        "roles": session.get("roles", []),
        "portal": session.get("portal", "admin"),
    }


@router.get("/admin/auth/expert-me", tags=["AdminAuth"])
def expert_me(
    x_admin_token: Annotated[Optional[str], Header()] = None,
    authorization: Annotated[Optional[str], Header()] = None,
):
    """Return current expert portal session identity."""
    session = resolve_expert_session(x_admin_token=x_admin_token, authorization=authorization)
    return {
        "username": session.get("username", "expert"),
        "roles": session.get("roles", []),
        "portal": session.get("portal", "expert"),
    }
