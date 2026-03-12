"""Governance, policy, and RBAC endpoints."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException

from ..api.deps import require_admin, require_admin_or_role
from ..roadmap_features import (
    activate_policy_version,
    create_policy_version,
    get_active_policy,
    get_tenant_budget,
    get_usage_summary,
    grant_role,
    list_audit_events,
    list_policy_versions,
    list_roles,
    log_audit_event,
    revoke_role,
    set_tenant_budget,
)

router = APIRouter()


@router.put("/admin/budgets/{tenant_id}", tags=["Governance"])
def upsert_tenant_budget(
    tenant_id: str,
    payload: Dict[str, Any],
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Create or update tenant budget limits."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["governance_admin", "platform_admin"],
        tenant_id=tenant_id,
    )
    daily = float(payload.get("daily_usd_limit", 0.0) or 0.0)
    monthly = float(payload.get("monthly_usd_limit", 0.0) or 0.0)
    enabled = bool(payload.get("enabled", True))
    set_tenant_budget(tenant_id=tenant_id, daily_usd_limit=daily, monthly_usd_limit=monthly, enabled=enabled)
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="budget_upsert",
        resource="tenant_budgets",
        tenant_id=tenant_id,
        metadata={"daily_usd_limit": daily, "monthly_usd_limit": monthly, "enabled": enabled, "roles": auth["roles"]},
    )
    return {"status": "updated", "budget": get_tenant_budget(tenant_id)}


@router.get("/admin/budgets/{tenant_id}", tags=["Governance"])
def get_budget(
    tenant_id: str,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Get tenant budget configuration."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["governance_viewer", "governance_admin", "platform_admin"],
        tenant_id=tenant_id,
    )
    return get_tenant_budget(tenant_id)


@router.get("/admin/quotas/usage", tags=["Governance"])
def get_quota_usage(
    tenant_id: Optional[str] = None,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Get usage summary for one or all tenants."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["governance_viewer", "governance_admin", "platform_admin"],
        tenant_id=tenant_id,
    )
    return get_usage_summary(tenant_id)


@router.get("/admin/audit/events", tags=["Governance"])
def get_audit_events(
    limit: int = 100,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Get latest audit events."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["audit_viewer", "platform_admin"],
    )
    return {"items": list_audit_events(limit=limit)}


@router.post("/admin/policies", tags=["Policy"])
def create_policy(
    payload: Dict[str, Any],
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Create or update a policy version."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["policy_admin", "platform_admin"],
    )
    version = str(payload.get("version") or "").strip()
    if not version:
        raise HTTPException(status_code=400, detail="version is required")
    description = str(payload.get("description") or "")
    config = payload.get("config") or {}
    if not isinstance(config, dict):
        raise HTTPException(status_code=400, detail="config must be an object")
    create_policy_version(version=version, config=config, description=description)
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="policy_upsert",
        resource="policy_versions",
        metadata={"version": version, "roles": auth["roles"]},
    )
    return {"status": "created_or_updated", "version": version}


@router.post("/admin/policies/{version}/activate", tags=["Policy"])
def activate_policy(
    version: str,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Activate one policy version."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["policy_admin", "platform_admin"],
    )
    if not activate_policy_version(version):
        raise HTTPException(status_code=404, detail=f"Policy not found: {version}")
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="policy_activate",
        resource="policy_versions",
        metadata={"version": version, "roles": auth["roles"]},
    )
    return {"status": "activated", "version": version}


@router.get("/admin/policies", tags=["Policy"])
def list_policies(
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """List policy versions."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["policy_viewer", "policy_admin", "platform_admin"],
    )
    return {"active": get_active_policy(), "items": list_policy_versions()}


@router.post("/admin/rbac/grants", tags=["Governance"])
def create_role_grant(payload: Dict[str, Any], x_admin_token: Optional[str] = Header(None)):
    """Grant a role to a user. Bootstrap is admin-token only."""
    require_admin(x_admin_token)
    user_id = str(payload.get("user_id") or "").strip()
    role_name = str(payload.get("role_name") or "").strip()
    tenant_id = payload.get("tenant_id")
    if not user_id or not role_name:
        raise HTTPException(status_code=400, detail="user_id and role_name are required")
    grant_role(user_id=user_id, role_name=role_name, tenant_id=str(tenant_id) if tenant_id else None)
    log_audit_event(
        actor="admin",
        action="rbac_grant",
        resource="rbac_user_roles",
        tenant_id=str(tenant_id) if tenant_id else None,
        metadata={"user_id": user_id, "role_name": role_name},
    )
    return {"status": "granted", "user_id": user_id, "role_name": role_name, "tenant_id": tenant_id}


@router.post("/admin/rbac/revokes", tags=["Governance"])
def delete_role_grant(payload: Dict[str, Any], x_admin_token: Optional[str] = Header(None)):
    """Revoke a role from a user. Bootstrap is admin-token only."""
    require_admin(x_admin_token)
    user_id = str(payload.get("user_id") or "").strip()
    role_name = str(payload.get("role_name") or "").strip()
    tenant_id = payload.get("tenant_id")
    if not user_id or not role_name:
        raise HTTPException(status_code=400, detail="user_id and role_name are required")
    removed = revoke_role(user_id=user_id, role_name=role_name, tenant_id=str(tenant_id) if tenant_id else None)
    log_audit_event(
        actor="admin",
        action="rbac_revoke",
        resource="rbac_user_roles",
        tenant_id=str(tenant_id) if tenant_id else None,
        metadata={"user_id": user_id, "role_name": role_name, "removed": removed},
    )
    return {"status": "revoked", "removed": removed}


@router.get("/admin/rbac/roles", tags=["Governance"])
def get_rbac_roles(user_id: Optional[str] = None, x_admin_token: Optional[str] = Header(None)):
    """List RBAC role bindings."""
    require_admin(x_admin_token)
    return {"items": list_roles(user_id=user_id)}
