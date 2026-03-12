"""Shared FastAPI authorization helpers."""

from __future__ import annotations

import secrets
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from ..roadmap_features import check_access
from ..settings_dynamic import settings


def require_admin(token: Optional[str]) -> None:
    """Authorize a request using the configured admin token."""
    configured = (settings.ADMIN_TOKEN or "").strip()
    previous = (settings.ADMIN_TOKEN_PREVIOUS or "").strip()

    ok = False
    if configured and token:
        ok = secrets.compare_digest(token, configured)
        if not ok and previous:
            ok = secrets.compare_digest(token, previous)

    if not ok:
        raise HTTPException(status_code=401, detail="Token inválido.")


def parse_header_roles(x_user_roles: Optional[str]) -> List[str]:
    """Parse comma-separated roles from header."""
    if not x_user_roles:
        return []
    return [role.strip() for role in str(x_user_roles).split(",") if role.strip()]


def require_admin_or_role(
    *,
    admin_token: Optional[str],
    user_id: Optional[str],
    user_roles_header: Optional[str],
    required_roles: List[str],
    tenant_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Authorize request by admin token or RBAC role."""
    try:
        require_admin(admin_token)
        return {"authorized_by": "admin_token", "roles": ["admin"]}
    except HTTPException:
        decision = check_access(
            user_id=user_id,
            tenant_id=tenant_id,
            required_roles=required_roles,
            header_roles=parse_header_roles(user_roles_header),
        )
        if decision.allowed:
            return {"authorized_by": "rbac", "roles": decision.roles}
    raise HTTPException(
        status_code=403,
        detail={"error": True, "message": "Acesso negado.", "required_roles": required_roles},
    )
