# Objective: API routes for human expert review portal.
"""Expert reviewer endpoints — profile, queue, assessments, kappa."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from ..api.deps import _roles_from_jwt, require_admin_or_role
from ..roadmap_features import log_audit_event
from ..schemas import (
    ExpertAccountCreateRequest,
    ExpertAccountUpdateRequest,
    ExpertAssessmentSubmitRequest,
    ExpertPreviewRequest,
    ExpertProfileUpdateRequest,
)
from ..services.expert_accounts import list_expert_accounts_public, register_expert_account, update_expert_account_admin
from ..services.expert_review import (
    ensure_expert_profile,
    expert_judge_agreement_report,
    get_next_review_item,
    list_available_themes,
    list_expert_assessments,
    submit_expert_assessment,
    update_expert_profile,
)

router = APIRouter()

_EXPERT_ROLES = ["expert_reviewer", "audit_viewer", "eval_admin", "researcher", "platform_admin"]
_ADMIN_MANAGE_ROLES = ["platform_admin", "eval_admin", "governance_admin", "admin"]


def _expert_id(x_user_id: Optional[str], auth: dict, authorization: Optional[str] = None) -> str:
    if x_user_id and str(x_user_id).strip():
        return str(x_user_id).strip()[:128]
    jwt_user, _, _ = _roles_from_jwt(authorization)
    if jwt_user:
        return str(jwt_user).strip()[:128]
    username = auth.get("username") or auth.get("authorized_by") or "anonymous"
    return str(username)[:128]


@router.get("/admin/experts/accounts", tags=["Experts"])
def list_expert_accounts_route(
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """List registered expert accounts (admin only)."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=_ADMIN_MANAGE_ROLES,
    )
    return {"items": list_expert_accounts_public()}


@router.post("/admin/experts/accounts", tags=["Experts"])
def create_expert_account_route(
    payload: ExpertAccountCreateRequest,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Register a new expert with name, phone, email and password."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=_ADMIN_MANAGE_ROLES,
    )
    try:
        account = register_expert_account(
            display_name=payload.display_name,
            email=payload.email,
            phone=payload.phone,
            password=payload.password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="expert_account_create",
        resource="expert_accounts",
        metadata={"email": account.get("email"), "roles": auth["roles"]},
    )
    return {"status": "created", "account": account}


@router.put("/admin/experts/accounts/{account_id}", tags=["Experts"])
def update_expert_account_route(
    account_id: int,
    payload: ExpertAccountUpdateRequest,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Update expert account fields or reset password."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=_ADMIN_MANAGE_ROLES,
    )
    try:
        account = update_expert_account_admin(
            account_id,
            display_name=payload.display_name,
            phone=payload.phone,
            password=payload.password,
            enabled=payload.enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not account:
        raise HTTPException(status_code=404, detail="Especialista não encontrado.")
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="expert_account_update",
        resource="expert_accounts",
        metadata={"account_id": account_id, "email": account.get("email"), "roles": auth["roles"]},
    )
    return {"status": "updated", "account": account}


@router.get("/admin/experts/themes", tags=["Experts"])
def get_expert_themes(
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """List benchmark themes available for expert area selection."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=_EXPERT_ROLES,
    )
    return {"items": list_available_themes()}


@router.get("/admin/experts/profile", tags=["Experts"])
def get_profile(
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Return the authenticated expert's profile."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=_EXPERT_ROLES,
    )
    expert_id = _expert_id(x_user_id, auth, authorization)
    return ensure_expert_profile(expert_id)


@router.put("/admin/experts/profile", tags=["Experts"])
def put_profile(
    payload: ExpertProfileUpdateRequest,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Update expert areas of expertise and display name."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=_EXPERT_ROLES,
    )
    expert_id = _expert_id(x_user_id, auth, authorization)
    profile = update_expert_profile(
        expert_id,
        display_name=payload.display_name,
        theme_ids=payload.theme_ids,
        credentials_note=payload.credentials_note,
    )
    log_audit_event(
        actor=expert_id,
        action="expert_profile_update",
        resource="expert_profiles",
        metadata={"theme_ids": payload.theme_ids, "roles": auth["roles"]},
    )
    return profile


@router.get("/admin/experts/next-item", tags=["Experts"])
def next_review_item(
    eval_run_id: Optional[str] = None,
    split: Optional[str] = "held_out",
    seed: Optional[int] = None,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Fetch the next catalog or eval item awaiting expert analysis."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=_EXPERT_ROLES,
    )
    expert_id = _expert_id(x_user_id, auth, authorization)
    item = get_next_review_item(expert_id, eval_run_id=eval_run_id, split=split, seed=seed)
    if not item:
        return {"status": "empty", "item": None}
    return {"status": "ok", "item": item}


@router.post("/admin/experts/assessments", tags=["Experts"])
def post_assessment(
    payload: ExpertAssessmentSubmitRequest,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Submit human expert analysis for one query."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=_EXPERT_ROLES,
    )
    expert_id = _expert_id(x_user_id, auth, authorization)
    profile = ensure_expert_profile(expert_id)
    allowed_themes = {str(t) for t in (profile.get("theme_ids") or [])}
    if allowed_themes and payload.theme not in allowed_themes:
        raise HTTPException(status_code=403, detail=f"Theme not in expert profile: {payload.theme}")

    saved = submit_expert_assessment(
        expert_id,
        benchmark_id=payload.benchmark_id,
        theme=payload.theme,
        query_text=payload.query_text,
        answer=payload.answer,
        reference=payload.reference,
        eval_run_id=payload.eval_run_id,
        judge_quality=payload.judge_quality,
        quality_score=payload.quality_score,
        rubric=payload.rubric.model_dump(),
        notes=payload.notes,
    )
    log_audit_event(
        actor=expert_id,
        action="expert_assessment_submit",
        resource="expert_assessments",
        metadata={
            "benchmark_id": payload.benchmark_id,
            "theme": payload.theme,
            "eval_run_id": payload.eval_run_id,
            "roles": auth["roles"],
        },
    )
    return {"status": "saved", "assessment": saved}


@router.get("/admin/experts/assessments", tags=["Experts"])
def get_assessments(
    theme: Optional[str] = None,
    eval_run_id: Optional[str] = None,
    limit: int = 100,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """List assessments by the authenticated expert."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=_EXPERT_ROLES,
    )
    expert_id = _expert_id(x_user_id, auth, authorization)
    return {"items": list_expert_assessments(expert_id=expert_id, theme=theme, eval_run_id=eval_run_id, limit=limit)}


@router.get("/admin/experts/metrics/kappa", tags=["Experts"])
def get_kappa_metrics(
    eval_run_id: Optional[str] = None,
    theme: Optional[str] = None,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Return judge vs human agreement (Cohen's kappa) metrics."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=["eval_viewer", "eval_admin", "researcher", "platform_admin"],
    )
    return expert_judge_agreement_report(eval_run_id=eval_run_id, theme=theme)


@router.post("/admin/experts/preview-answer", tags=["Experts"])
async def preview_system_answer(
    payload: ExpertPreviewRequest,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Execute one query through the router so experts can score the live answer."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        authorization=authorization,
        required_roles=_EXPERT_ROLES,
    )
    from ..schemas import QueryRequest, WorkloadHints
    from ..services.query_runtime import process_query_request

    hints = None
    if payload.theme or payload.benchmark_id:
        hints = WorkloadHints(theme=payload.theme, benchmark_id=payload.benchmark_id)
    req = QueryRequest(
        query=payload.query,
        workload_hints=hints,
        use_cache=False,
    )
    wrapped = await process_query_request(req)
    resp = wrapped.get("result", wrapped)
    meta = resp.get("metadata") or {}
    return {
        "answer": str(resp.get("answer") or ""),
        "model": resp.get("model"),
        "judge_quality": float(meta.get("quality", 0.0) or 0.0),
        "latency_s": resp.get("latency_s"),
        "metadata": meta,
        "reviewer": _expert_id(x_user_id, auth, authorization),
    }
