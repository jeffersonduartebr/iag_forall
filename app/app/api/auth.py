# Objective: API authentication helpers (API keys, JWT, job ownership).
"""Authentication utilities for public API endpoints."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import Header, HTTPException, Request

from ..config.secrets_redaction import redact_secrets
from ..settings_dynamic import settings

logger = logging.getLogger(__name__)

__all__ = [
    "AuthContext",
    "optional_api_auth",
    "redact_secrets",
    "require_api_auth",
    "resolve_auth",
    "verify_job_access",
]


@dataclass
class AuthContext:
    """Resolved caller identity for authorization and job ownership."""

    authenticated: bool = False
    method: str = "anonymous"
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    api_key_hint: Optional[str] = None

    @property
    def owner_key(self) -> str:
        """Stable ownership key stored with async query jobs."""
        if self.tenant_id:
            return f"tenant:{self.tenant_id}"
        if self.user_id:
            return f"user:{self.user_id}"
        if self.api_key_hint:
            return f"key:{self.api_key_hint}"
        return "anonymous"

    def owner_hash(self) -> str:
        """Hash of owner_key for Redis job records."""
        return hashlib.sha256(self.owner_key.encode("utf-8")).hexdigest()[:32]


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _encode_jwt_hs256(payload: Dict[str, Any], secret: str) -> str:
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode("utf-8"))
    body = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header}.{body}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header}.{body}.{_b64url_encode(signature)}"


def _decode_jwt_hs256(token: str, secret: str) -> Dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("invalid_jwt_format")
    header_b64, payload_b64, signature_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual = _b64url_decode(signature_b64)
    if not hmac.compare_digest(expected, actual):
        raise ValueError("invalid_jwt_signature")
    payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    exp = payload.get("exp")
    if exp is not None and time.time() > float(exp):
        raise ValueError("jwt_expired")
    return payload


def _parse_api_keys() -> List[str]:
    raw = settings.get("API_KEYS", "") or ""
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _auth_from_api_key(token: str) -> Optional[AuthContext]:
    configured = _parse_api_keys()
    admin = (settings.ADMIN_TOKEN or "").strip()
    previous = (settings.ADMIN_TOKEN_PREVIOUS or "").strip()
    for candidate in configured:
        if candidate and secrets.compare_digest(token, candidate):
            return AuthContext(
                authenticated=True,
                method="api_key",
                api_key_hint=hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:12],
            )
    if admin and secrets.compare_digest(token, admin):
        return AuthContext(authenticated=True, method="admin_token", roles=["admin"])
    if previous and secrets.compare_digest(token, previous):
        return AuthContext(authenticated=True, method="admin_token", roles=["admin"])
    return None


def _auth_from_jwt(token: str) -> Optional[AuthContext]:
    secret = (settings.get("JWT_SECRET", "") or "").strip()
    if not secret:
        return None
    try:
        claims = _decode_jwt_hs256(token, secret)
    except ValueError as exc:
        logger.debug("[auth] JWT rejected: %s", exc)
        return None
    roles_raw = claims.get("roles") or claims.get("role") or []
    if isinstance(roles_raw, str):
        roles = [roles_raw]
    else:
        roles = [str(role) for role in roles_raw]
    return AuthContext(
        authenticated=True,
        method="jwt",
        user_id=str(claims.get("sub") or claims.get("user_id") or "") or None,
        tenant_id=str(claims.get("tenant_id") or "") or None,
        roles=roles,
    )


def resolve_auth(
    *,
    authorization: Optional[str] = None,
    x_api_key: Optional[str] = None,
    required: bool = False,
) -> AuthContext:
    """Resolve caller identity from API key or JWT bearer token."""
    token = (x_api_key or "").strip() or _extract_bearer_token(authorization)
    if token:
        ctx = _auth_from_jwt(token)
        if ctx is None:
            ctx = _auth_from_api_key(token)
        if ctx is not None:
            return ctx
        if required:
            raise HTTPException(status_code=401, detail="Token inválido ou expirado.")
    if required:
        raise HTTPException(status_code=401, detail="Autenticação obrigatória.")
    return AuthContext(authenticated=False, method="anonymous")


def require_api_auth(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
) -> AuthContext:
    """FastAPI dependency enforcing API authentication when enabled."""
    required = settings._get_bool("REQUIRE_API_AUTH", False)
    ctx = resolve_auth(authorization=authorization, x_api_key=x_api_key, required=required)
    request.state.auth = ctx
    return ctx


def optional_api_auth(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
) -> AuthContext:
    """FastAPI dependency that resolves auth without requiring it."""
    ctx = resolve_auth(authorization=authorization, x_api_key=x_api_key, required=False)
    request.state.auth = ctx
    return ctx


def verify_job_access(record: Dict[str, Any], auth: AuthContext) -> None:
    """Ensure the caller may access one queued query job."""
    if not settings._get_bool("REQUIRE_API_AUTH", False):
        return
    expected = str(record.get("owner_hash") or "").strip()
    if not expected:
        raise HTTPException(status_code=403, detail={"error": True, "message": "Job sem ownership registrado."})
    if not auth.authenticated or auth.owner_hash() != expected:
        raise HTTPException(status_code=403, detail={"error": True, "message": "Acesso negado ao job."})
