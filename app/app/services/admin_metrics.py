# Objective: Admin dashboard metrics (Prometheus proxy + health snapshot).
"""Aggregate operational metrics for the admin web console."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

from ..health import get_full_health_check
from ..reliability import get_cascade_detector, get_circuit_breaker_manager
from ..roadmap_features import get_usage_summary

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090").rstrip("/")

CHART_QUERIES: Dict[str, str] = {
    "qps": "sum(rate(api_requests_total[1m]))",
    "latency_p95": "histogram_quantile(0.95, sum(rate(api_request_latency_seconds_bucket[5m])) by (le))",
    "provider_cost": "sum(increase(providers_cost_usd[5m]))",
    "cache_hit_rate": (
        "sum(rate(semantic_cache_hits_total[5m])) / "
        "clamp_min(sum(rate(semantic_cache_lookups_total[5m])), 1)"
    ),
    "chosen_models": "topk(5, sum by (model) (increase(router_chosen_model_total[5m])))",
}


async def _prom_query(query: str) -> Optional[List[Dict[str, Any]]]:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query})
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data.get("status") != "success":
                return None
            return data.get("data", {}).get("result") or []
    except Exception:
        return None


async def _prom_query_range(query: str, *, start: float, end: float, step: str = "5s") -> Optional[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{PROMETHEUS_URL}/api/v1/query_range",
                params={"query": query, "start": start, "end": end, "step": step},
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data.get("status") != "success":
                return None
            return data.get("data")
    except Exception:
        return None


async def build_dashboard_summary() -> Dict[str, Any]:
    """Build one-shot dashboard cards."""
    health = await get_full_health_check()
    breakers_list = get_circuit_breaker_manager().get_all_statuses()
    open_breakers = [b.get("model") for b in breakers_list if b.get("state") == "open"]
    cascade = get_cascade_detector().get_status()
    usage = get_usage_summary(None)

    outcomes = await _prom_query('sum(increase(router_query_outcome_total[5m])) by (outcome)') or []

    return {
        "timestamp": time.time(),
        "health": health,
        "circuit_breakers": {
            "open": open_breakers,
            "total": len(breakers_list),
        },
        "cascade": cascade,
        "usage": usage,
        "query_outcomes_5m": outcomes,
        "prometheus_available": bool(await _prom_query("up")),
    }


async def build_dashboard_series(*, window_s: int = 3600, step: str = "5s") -> Dict[str, Any]:
    """Return time-series for dashboard charts."""
    end = time.time()
    start = end - max(60, window_s)
    series: Dict[str, Any] = {}
    for key, query in CHART_QUERIES.items():
        series[key] = await _prom_query_range(query, start=start, end=end, step=step)
    return {"window_s": window_s, "step": step, "series": series}
