# Objective: API layer code for eval routes.
"""Academic and evaluation endpoints."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from celery.result import AsyncResult
from fastapi import APIRouter, Body, Header, HTTPException

from ..api.deps import require_admin_or_role
from ..celery_app import celery_app
from ..golden_sets import evaluate_golden_set_gate, get_golden_set, get_golden_set_prompts, list_golden_sets
from ..roadmap_features import (
    create_eval_run,
    eval_significance_report,
    get_eval_run,
    list_eval_run_results,
    list_eval_runs,
    log_audit_event,
    update_eval_run_status,
)
from ..schemas import EvalRunCreateRequest, EvalRunExecuteRequest
from ..settings_dynamic import settings
from ..tasks import task_execute_eval_run

router = APIRouter()


@router.post("/admin/evals/runs", tags=["Eval"])
def create_eval(
    payload: EvalRunCreateRequest,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Create an eval run (MVP academic harness)."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_admin", "researcher", "platform_admin"],
        tenant_id=str(payload.tenant_id) if payload.tenant_id else None,
    )
    prompts = list(payload.prompts or [])
    golden_set_id = str(payload.golden_set_id or "").strip() or None
    if golden_set_id and not prompts:
        prompts = get_golden_set_prompts(golden_set_id)
    if not isinstance(prompts, list) or not prompts:
        raise HTTPException(status_code=400, detail="prompts must be a non-empty list")
    prompts = [str(prompt) for prompt in prompts if str(prompt).strip()]
    run_id = payload.run_id or f"eval_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    policy_version = payload.policy_version
    tenant_id = payload.tenant_id
    notes = str(payload.notes or "")
    create_eval_run(
        run_id=run_id,
        prompts=prompts,
        policy_version=str(policy_version) if policy_version else None,
        tenant_id=str(tenant_id) if tenant_id else None,
        notes=notes,
        metadata={"golden_set_id": golden_set_id} if golden_set_id else {},
    )
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="eval_create",
        resource="eval_runs",
        tenant_id=tenant_id,
        metadata={"run_id": run_id, "prompt_count": len(prompts), "roles": auth["roles"], "golden_set_id": golden_set_id},
    )
    return {"status": "queued", "run_id": run_id, "prompt_count": len(prompts)}


@router.post("/admin/evals/runs/{run_id}/execute", tags=["Eval"])
def execute_eval(
    run_id: str,
    payload: EvalRunExecuteRequest = Body(default=EvalRunExecuteRequest()),
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Enqueue asynchronous eval execution in Celery."""
    run = get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run not found: {run_id}")
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_admin", "researcher", "platform_admin"],
        tenant_id=run.get("tenant_id"),
    )
    task = task_execute_eval_run.delay(
        run_id=run_id,
        modality=str(payload.modality or "text"),
        use_cache=bool(payload.use_cache),
        max_tokens=int(payload.max_tokens or settings.MAX_TOKENS_DEFAULT),
        temperature=float(payload.temperature or settings.TEMPERATURE_DEFAULT),
    )
    update_eval_run_status(
        run_id,
        "queued",
        {
            "queued_at": time.time(),
            "task_id": task.id,
            "modality": str(payload.modality or "text"),
            "use_cache": bool(payload.use_cache),
            "max_tokens": int(payload.max_tokens or settings.MAX_TOKENS_DEFAULT),
            "temperature": float(payload.temperature or settings.TEMPERATURE_DEFAULT),
        },
    )
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="eval_execute_queued",
        resource="eval_runs",
        tenant_id=run.get("tenant_id"),
        metadata={"run_id": run_id, "task_id": task.id, "roles": auth["roles"]},
    )
    return {"status": "queued", "run_id": run_id, "task_id": task.id}


@router.get("/admin/evals/runs/{run_id}", tags=["Eval"])
def get_eval(
    run_id: str,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Get eval run details."""
    run = get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run not found: {run_id}")
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_viewer", "eval_admin", "researcher", "platform_admin"],
        tenant_id=run.get("tenant_id"),
    )
    return run


@router.get("/admin/evals/runs", tags=["Eval"])
def list_evals(
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """List eval runs."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_viewer", "eval_admin", "researcher", "platform_admin"],
    )
    return {"items": list_eval_runs()}


@router.get("/admin/evals/runs/{run_id}/results", tags=["Eval"])
def get_eval_results(
    run_id: str,
    limit: int = 2000,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Get individual result rows for one eval run."""
    run = get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run not found: {run_id}")
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_viewer", "eval_admin", "researcher", "platform_admin"],
        tenant_id=run.get("tenant_id"),
    )
    return {"run_id": run_id, "items": list_eval_run_results(run_id, limit=limit)}


@router.get("/admin/evals/runs/{run_id}/significance", tags=["Eval"])
def get_eval_significance(
    run_id: str,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Get significance report for model comparisons in one eval run."""
    run = get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run not found: {run_id}")
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_viewer", "eval_admin", "researcher", "platform_admin"],
        tenant_id=run.get("tenant_id"),
    )
    return eval_significance_report(run_id)


@router.get("/admin/evals/golden-sets", tags=["Eval"])
def get_builtin_golden_sets(
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """List built-in golden sets available for repeatable benchmark runs."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_viewer", "eval_admin", "researcher", "platform_admin"],
    )
    return {"items": list_golden_sets()}


@router.get("/admin/evals/golden-sets/{golden_set_id}", tags=["Eval"])
def get_builtin_golden_set(
    golden_set_id: str,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Return one built-in golden set with prompt items and gate thresholds."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_viewer", "eval_admin", "researcher", "platform_admin"],
    )
    golden_set = get_golden_set(golden_set_id)
    if not golden_set:
        raise HTTPException(status_code=404, detail=f"Golden set not found: {golden_set_id}")
    return golden_set


@router.get("/admin/evals/runs/{run_id}/gate-report", tags=["Eval"])
def get_eval_gate_report(
    run_id: str,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Evaluate one run against the gates attached to its golden set, when configured."""
    run = get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run not found: {run_id}")
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_viewer", "eval_admin", "researcher", "platform_admin"],
        tenant_id=run.get("tenant_id"),
    )
    metadata = run.get("metadata") or {}
    golden_set_id = metadata.get("golden_set_id")
    if not golden_set_id:
        return {"run_id": run_id, "gate_report": {"passed": True, "checks": {}, "metrics": {}, "gates": {}}}
    golden_set = get_golden_set(str(golden_set_id))
    if not golden_set:
        raise HTTPException(status_code=404, detail=f"Golden set not found: {golden_set_id}")
    return {
        "run_id": run_id,
        "golden_set_id": golden_set_id,
        "gate_report": evaluate_golden_set_gate(list_eval_run_results(run_id, limit=5000), golden_set.get("gates") or {}),
    }


@router.get("/admin/evals/tasks/{task_id}", tags=["Eval"])
def get_eval_task_status(
    task_id: str,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Get Celery task status/result for eval execution."""
    require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_viewer", "eval_admin", "researcher", "platform_admin"],
    )
    task = AsyncResult(task_id, app=celery_app)
    out: Dict[str, Any] = {
        "task_id": task_id,
        "state": task.state,
        "ready": bool(task.ready()),
        "successful": bool(task.successful()) if task.ready() else False,
    }
    if task.ready():
        try:
            out["result"] = task.result
        except Exception as exc:
            out["result_error"] = str(exc)
    return out


@router.post("/admin/evals/tasks/{task_id}/cancel", tags=["Eval"])
def cancel_eval_task(
    task_id: str,
    terminate: bool = False,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Cancel/revoke a queued eval Celery task."""
    auth = require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_admin", "platform_admin"],
    )
    celery_app.control.revoke(task_id, terminate=terminate)
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="eval_task_cancel",
        resource="celery_task",
        metadata={"task_id": task_id, "terminate": terminate, "roles": auth["roles"]},
    )
    return {"status": "revoked", "task_id": task_id, "terminate": terminate}
