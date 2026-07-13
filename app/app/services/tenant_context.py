# Objective: Resolve and enforce tenant identity for multi-tenant production.
"""Tenant binding helpers for authenticated query requests."""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException

from ..api.auth import AuthContext
from ..schemas import QueryRequest
from ..settings_dynamic import settings


def _enforce_binding() -> bool:
    return str(settings.get("ENFORCE_TENANT_BINDING", "1")).strip().lower() in {"1", "true", "yes", "on"}


def resolve_effective_tenant_id(
    req: QueryRequest,
    auth: Optional[AuthContext] = None,
) -> Optional[str]:
    """Resolve tenant from auth context, falling back to request body when allowed."""
    auth_tenant = (auth.tenant_id or "").strip() if auth and auth.authenticated else ""
    body_tenant = (req.tenant_id or "").strip() if getattr(req, "tenant_id", None) else ""

    if auth_tenant:
        if body_tenant and body_tenant != auth_tenant and _enforce_binding():
            raise HTTPException(
                status_code=403,
                detail={
                    "error": True,
                    "category": "tenant_mismatch",
                    "message": "tenant_id do payload não corresponde ao token.",
                    "expected": auth_tenant,
                },
            )
        return auth_tenant

    if body_tenant:
        return body_tenant

    if auth and auth.authenticated and auth.user_id:
        return f"user:{auth.user_id}"

    return None


def bind_tenant_to_request(req: QueryRequest, auth: Optional[AuthContext] = None) -> QueryRequest:
    """Attach resolved tenant_id to the request object."""
    tenant_id = resolve_effective_tenant_id(req, auth)
    if tenant_id:
        req.tenant_id = tenant_id
    return req
