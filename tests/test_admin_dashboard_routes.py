# Objective: Test coverage for admin dashboard API routes.
"""Tests for admin dashboard summary and metrics series."""

import pytest
from app.api import admin_dashboard_routes as routes


@pytest.mark.asyncio
async def test_dashboard_summary(monkeypatch):
    """Summary endpoint should delegate to admin_metrics service."""
    monkeypatch.setattr(routes, "resolve_admin_session", lambda **kwargs: {"username": "admin"})
    monkeypatch.setattr(
        routes,
        "build_dashboard_summary",
        lambda: _async_return({"health": {"status": "ok"}, "prometheus_available": True}),
    )
    out = await routes.dashboard_summary()
    assert out["health"]["status"] == "ok"
    assert out["prometheus_available"] is True


@pytest.mark.asyncio
async def test_dashboard_series(monkeypatch):
    """Series endpoint should return chart payload."""
    monkeypatch.setattr(routes, "resolve_admin_session", lambda **kwargs: {"username": "admin"})
    monkeypatch.setattr(
        routes,
        "build_dashboard_series",
        lambda **kwargs: _async_return({"window_s": 1800, "step": "5s", "series": {"qps": {}}}),
    )
    out = await routes.dashboard_series(window_s=1800, step="5s")
    assert out["window_s"] == 1800
    assert "qps" in out["series"]


@pytest.mark.asyncio
async def test_dashboard_roi(monkeypatch):
    """ROI endpoint should delegate to roi_analytics service."""
    monkeypatch.setattr(routes, "resolve_admin_session", lambda **kwargs: {"username": "admin"})
    monkeypatch.setattr(
        routes,
        "build_roi_report",
        lambda **kwargs: {
            "insufficient_data": False,
            "summary": {"savings_usd": 12.5, "savings_pct": 40.0, "query_count": 10},
            "daily_series": [],
            "model_breakdown": [],
        },
    )
    out = await routes.dashboard_roi(days=30, baseline_model="openai/gpt-4o")
    assert out["summary"]["savings_usd"] == 12.5
    assert out["summary"]["query_count"] == 10


@pytest.mark.asyncio
async def test_dashboard_requires_auth(monkeypatch):
    """Unauthorized requests should be rejected."""
    from fastapi import HTTPException

    def _deny(**kwargs):
        raise HTTPException(status_code=401, detail="Não autorizado.")

    monkeypatch.setattr(routes, "resolve_admin_session", _deny)
    with pytest.raises(HTTPException) as exc:
        await routes.dashboard_summary()
    assert exc.value.status_code == 401


async def _async_return(value):
    return value
