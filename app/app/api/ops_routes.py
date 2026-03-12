"""Operational endpoints such as health checks."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..health import get_full_health_check, get_liveness_check, get_readiness_check
from ..observability import COMPONENT_HEALTH

router = APIRouter()


@router.get("/health", tags=["Ops"])
async def health():
    """Deep health check with all component statuses."""
    result = await get_full_health_check()

    for name, component in result.get("components", {}).items():
        COMPONENT_HEALTH.labels(component=name).set(1 if component.get("healthy") else 0)

    status_code = 200 if result["status"] == "healthy" else (
        503 if result["status"] == "unhealthy" else 200
    )
    return JSONResponse(content=result, status_code=status_code)


@router.get("/healthz", tags=["Ops"])
async def liveness():
    """Kubernetes liveness probe - is the app running?"""
    return await get_liveness_check()


@router.get("/ready", tags=["Ops"])
async def readiness():
    """Kubernetes readiness probe - is the app ready to serve traffic?"""
    result = await get_readiness_check()
    status_code = 200 if result["status"] == "ready" else 503
    return JSONResponse(content=result, status_code=status_code)
