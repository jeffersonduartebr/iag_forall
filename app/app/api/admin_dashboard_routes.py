# Objective: Admin dashboard API routes.
"""Dashboard summary and metrics series for admin UI."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Header, Query
from fastapi.responses import StreamingResponse

from ..api.admin_auth_routes import resolve_admin_session
from ..services.admin_logs import DEFAULT_QUERY, stream_logs
from ..services.admin_metrics import build_dashboard_series, build_dashboard_summary
from ..services.expert_review import build_expert_kappa_dashboard
from ..services.roi_analytics import build_roi_report

router = APIRouter()


@router.get("/admin/dashboard/summary", tags=["AdminDashboard"])
async def dashboard_summary(
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Return operational snapshot for dashboard cards."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    return await build_dashboard_summary()


@router.get("/admin/dashboard/series", tags=["AdminDashboard"])
async def dashboard_series(
    window_s: int = Query(3600, ge=60, le=86400),
    step: str = Query("5s"),
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Return Prometheus time-series for dashboard charts."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    return await build_dashboard_series(window_s=window_s, step=step)


@router.get("/admin/dashboard/expert-kappa", tags=["AdminDashboard"])
async def dashboard_expert_kappa(
    eval_run_id: Optional[str] = None,
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Return judge vs human kappa metrics by theme for the admin dashboard."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    return build_expert_kappa_dashboard(eval_run_id=eval_run_id)


@router.get("/admin/dashboard/roi", tags=["AdminDashboard"])
async def dashboard_roi(
    tenant_id: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    baseline_model: Optional[str] = None,
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """ROI report: actual router cost vs premium baseline (contrafactual)."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    return build_roi_report(tenant_id=tenant_id, days=days, baseline_model=baseline_model)


@router.get("/admin/logs/stream", tags=["AdminDashboard"])
async def logs_stream(
    query: str = Query(DEFAULT_QUERY),
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Stream logs via Server-Sent Events."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)

    async def _event_generator():
        async for entry in stream_logs(query=query):
            yield f"data: {json.dumps(entry, default=str)}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
