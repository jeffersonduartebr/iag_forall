# Objective: API layer code for governance routes.
"""Governance, policy, and RBAC endpoints."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from ..api.deps import require_admin, require_admin_or_role
from ..roadmap_features import (
    activate_policy_version,
    check_tenant_budget,
    create_policy_version,
    get_active_policy,
    get_tenant_budget,
    get_usage_summary,
    grant_role,
    list_audit_events,
    list_policy_versions,
    list_response_reviews,
    list_roles,
    list_tenant_budgets,
    log_audit_event,
    revoke_role,
    set_tenant_budget,
    update_response_review,
)
from ..schemas import (
    PolicyCreateRequest,
    ResponseReviewUpdateRequest,
    RoleGrantRequest,
    RoleRevokeRequest,
    TenantBudgetUpdateRequest,
)
from ..services.governance_runtime import invalidate_runtime_policy_cache_async

router = APIRouter()


@router.get("/admin/budgets", tags=["Governance"])
def list_budgets(
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """List all tenant budgets."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=["governance_viewer", "governance_admin", "platform_admin"],
    )
    return {"items": list_tenant_budgets()}


@router.get("/admin/budgets/{tenant_id}/check", tags=["Governance"])
def check_budget(
    tenant_id: str,
    projected_cost_usd: float = 0.0,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Preview budget decision for a tenant."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=["governance_viewer", "governance_admin", "platform_admin"],
        tenant_id=tenant_id,
    )
    result = check_tenant_budget(tenant_id, projected_cost_usd=projected_cost_usd)
    return {
        "allowed": result.allowed,
        "reason": result.reason,
        "daily_spent": result.daily_spent,
        "monthly_spent": result.monthly_spent,
        "daily_limit": result.daily_limit,
        "monthly_limit": result.monthly_limit,
    }


@router.put("/admin/budgets/{tenant_id}", tags=["Governance"])
def upsert_tenant_budget(
    tenant_id: str,
    payload: TenantBudgetUpdateRequest,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Create or update tenant budget limits."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=["governance_admin", "platform_admin"],
        tenant_id=tenant_id,
    )
    daily = float(payload.daily_usd_limit or 0.0)
    monthly = float(payload.monthly_usd_limit or 0.0)
    enabled = bool(payload.enabled)
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
    authorization: Optional[str] = Header(None),
):
    """Get tenant budget configuration."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
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
    authorization: Optional[str] = Header(None),
):
    """Get usage summary for one or all tenants."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
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
    authorization: Optional[str] = Header(None),
):
    """Get latest audit events."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=["audit_viewer", "platform_admin"],
    )
    return {"items": list_audit_events(limit=limit)}


@router.post("/admin/policies", tags=["Policy"])
def create_policy(
    payload: PolicyCreateRequest,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Create or update a policy version."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=["policy_admin", "platform_admin"],
    )
    version = str(payload.version).strip()
    description = str(payload.description or "")
    config = dict(payload.config or {})
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
    authorization: Optional[str] = Header(None),
):
    """Activate one policy version."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=["policy_admin", "platform_admin"],
    )
    if not activate_policy_version(version):
        raise HTTPException(status_code=404, detail=f"Policy not found: {version}")
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(invalidate_runtime_policy_cache_async())
    except RuntimeError:
        asyncio.run(invalidate_runtime_policy_cache_async())
    except Exception:
        pass
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
    authorization: Optional[str] = Header(None),
):
    """List policy versions."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=["policy_viewer", "policy_admin", "platform_admin"],
    )
    return {"active": get_active_policy(), "items": list_policy_versions()}


@router.post("/admin/rbac/grants", tags=["Governance"])
def create_role_grant(payload: RoleGrantRequest, x_admin_token: Optional[str] = Header(None)):
    """Grant a role to a user. Bootstrap is admin-token only."""
    require_admin(x_admin_token)
    user_id = str(payload.user_id).strip()
    role_name = str(payload.role_name).strip()
    tenant_id = payload.tenant_id
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
def delete_role_grant(payload: RoleRevokeRequest, x_admin_token: Optional[str] = Header(None)):
    """Revoke a role from a user. Bootstrap is admin-token only."""
    require_admin(x_admin_token)
    user_id = str(payload.user_id).strip()
    role_name = str(payload.role_name).strip()
    tenant_id = payload.tenant_id
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


@router.get("/admin/reviews", tags=["Governance"])
def get_response_reviews(
    status: Optional[str] = None,
    limit: int = 100,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """List response-review items queued for human follow-up."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=["audit_viewer", "platform_admin"],
    )
    return {"items": list_response_reviews(status=status, limit=limit)}


@router.post("/admin/reviews/{review_id}", tags=["Governance"])
def apply_response_review(
    review_id: int,
    payload: ResponseReviewUpdateRequest,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Record one reviewer decision for an answer awaiting human validation."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=["audit_viewer", "platform_admin"],
    )
    review_status = str(getattr(payload.review_status, "value", payload.review_status) or "").strip()
    if review_status not in {"reviewed", "rejected"}:
        raise HTTPException(status_code=400, detail="review_status must be reviewed or rejected")
    updated = update_response_review(
        review_id,
        review_status=review_status,
        reviewer_id=x_user_id or auth["authorized_by"],
        reviewer_notes=str(payload.reviewer_notes or "") or None,
        corrected_answer=str(payload.corrected_answer or "") or None,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Review item not found: {review_id}")
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="response_review_update",
        resource="response_reviews",
        metadata={"review_id": review_id, "review_status": review_status, "roles": auth["roles"]},
    )
    return {"status": "updated", "review_id": review_id, "review_status": review_status}
