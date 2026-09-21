# Objective: Admin dashboard API routes.
"""Dashboard summary and metrics series for admin UI."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from ..services.admin_logs import DEFAULT_QUERY, stream_logs
from ..services.admin_metrics import build_dashboard_series, build_dashboard_summary
from ..services.expert_review import build_expert_kappa_dashboard
from ..services.roi_analytics import build_roi_report
from .dependencies import admin_session

router = APIRouter(
    # Ao nível do router, não por handler: uma rota acrescentada aqui
    # amanhã fica protegida sem ninguém se lembrar disso.
    dependencies=[Depends(admin_session)],
)


@router.get("/admin/dashboard/summary", tags=["AdminDashboard"])
async def dashboard_summary():
    """Return operational snapshot for dashboard cards."""
    return await build_dashboard_summary()


@router.get("/admin/dashboard/series", tags=["AdminDashboard"])
async def dashboard_series(
    window_s: int = Query(3600, ge=60, le=86400),
    # `step` ia sem validação para o `query_range` do Prometheus. Não é uma
    # expressão PromQL — as consultas são constantes — mas `step=1s` sobre 24 h
    # pede 86 400 pontos por série, cinco séries de cada vez, e é o chamador que
    # escolhe. O padrão fixa a forma e o servidor impõe o mínimo.
    step: str = Query("5s", pattern=r"^[1-9][0-9]{0,4}(ms|s|m|h)$"),
):
    """Return Prometheus time-series for dashboard charts."""
    return await build_dashboard_series(window_s=window_s, step=step)


@router.get("/admin/dashboard/expert-kappa", tags=["AdminDashboard"])
async def dashboard_expert_kappa(
    eval_run_id: Optional[str] = None,
):
    """Return judge vs human kappa metrics by theme for the admin dashboard."""
    return build_expert_kappa_dashboard(eval_run_id=eval_run_id)


@router.get("/admin/dashboard/roi", tags=["AdminDashboard"])
async def dashboard_roi(
    tenant_id: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    baseline_model: Optional[str] = None,
):
    """ROI report: actual router cost vs premium baseline (contrafactual)."""
    return build_roi_report(tenant_id=tenant_id, days=days, baseline_model=baseline_model)


@router.get("/admin/logs/stream", tags=["AdminDashboard"])
async def logs_stream(
    query: str = Query(DEFAULT_QUERY),
):
    """Stream logs via Server-Sent Events."""

    async def _event_generator():
        async for entry in stream_logs(query=query):
            yield f"data: {json.dumps(entry, default=str)}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
