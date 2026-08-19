# Objective: API layer code for deps.
"""Shared FastAPI authorization helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from ..roadmap_features import check_access
from ..settings_dynamic import settings
from .auth import _auth_from_jwt, _extract_bearer_token


def require_admin(token: Optional[str] = None, authorization: Optional[str] = None) -> None:
    """Authorize a request using admin token or admin JWT session."""
    from .admin_auth_routes import resolve_admin_session

    try:
        resolve_admin_session(x_admin_token=token, authorization=authorization)
    except HTTPException as exc:
        raise HTTPException(status_code=401, detail=exc.detail) from exc


def parse_header_roles(x_user_roles: Optional[str]) -> List[str]:
    """Parse comma-separated roles from header (deprecated unless explicitly trusted)."""
    if not x_user_roles:
        return []
    return [role.strip() for role in str(x_user_roles).split(",") if role.strip()]


def _roles_from_jwt(authorization: Optional[str]) -> tuple[Optional[str], Optional[str], List[str]]:
    """Extract user_id, tenant_id and roles from a JWT bearer token."""
    token = _extract_bearer_token(authorization)
    if not token:
        return None, None, []
    ctx = _auth_from_jwt(token)
    if ctx is None:
        return None, None, []
    return ctx.user_id, ctx.tenant_id, list(ctx.roles)


def require_admin_or_role(
    *,
    admin_token: Optional[str],
    user_id: Optional[str],
    user_roles_header: Optional[str],
    required_roles: List[str],
    tenant_id: Optional[str] = None,
    authorization: Optional[str] = None,
) -> Dict[str, Any]:
    """Authorize request by admin token, JWT, or DB-backed RBAC."""
    try:
        require_admin(admin_token, authorization)
        return {"authorized_by": "admin_token", "roles": ["admin"]}
    except HTTPException:
        pass

    jwt_user, jwt_tenant, jwt_roles = _roles_from_jwt(authorization)
    effective_user = jwt_user or user_id
    effective_tenant = jwt_tenant or tenant_id

    header_roles: Optional[List[str]] = None
    if settings._get_bool("TRUST_HEADER_ROLES", False):
        header_roles = parse_header_roles(user_roles_header)

    decision = check_access(
        user_id=effective_user,
        tenant_id=effective_tenant,
        required_roles=required_roles,
        header_roles=header_roles,
        jwt_roles=jwt_roles or None,
    )
    if decision.allowed:
        return {"authorized_by": decision.reason, "roles": decision.roles}
    raise HTTPException(
        status_code=403,
        detail={"error": True, "message": "Acesso negado.", "required_roles": required_roles},
    )
