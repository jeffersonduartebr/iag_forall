# Objective: ROI and cost-savings analytics for customer-facing dashboards.
"""Compute router savings vs a configurable premium baseline model."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from app.db import get_engine
from app.model_registry import get_model_config
from app.query_service import ensure_query_log
from app.roadmap_features import get_usage_summary

DEFAULT_BASELINE_MODEL = "openai/gpt-4o"
DEFAULT_QUALITY_THRESHOLD = 6.0
VISION_BASELINE_SURCHARGE_USD = 0.004


def _baseline_model_name(override: Optional[str] = None) -> str:
    raw = (override or os.getenv("ROI_BASELINE_MODEL", DEFAULT_BASELINE_MODEL) or DEFAULT_BASELINE_MODEL).strip()
    return raw


def _quality_threshold() -> float:
    try:
        return float(os.getenv("ROI_QUALITY_THRESHOLD", str(DEFAULT_QUALITY_THRESHOLD)))
    except ValueError:
        return DEFAULT_QUALITY_THRESHOLD


def _estimate_tokens(text_value: Optional[str]) -> int:
    if not text_value:
        return 0
    return max(1, int(len(str(text_value)) / 4))


def _baseline_unit_cost(
    *,
    query_text: str,
    answer: str,
    modality: str,
    baseline_model: str,
) -> float:
    cfg = get_model_config(baseline_model)
    if cfg is None:
        # Fallback pricing similar to GPT-4 class when registry lookup fails
        tokens_in = _estimate_tokens(query_text)
        tokens_out = _estimate_tokens(answer)
        cost = (tokens_in / 1000) * 0.005 + (tokens_out / 1000) * 0.015
    else:
        cost = cfg.calculate_cost(_estimate_tokens(query_text), _estimate_tokens(answer))
    if str(modality or "").lower() in {"vision", "multimodal"}:
        cost += VISION_BASELINE_SURCHARGE_USD
    return float(cost)


def _row_actual_cost(row: Dict[str, Any]) -> float:
    for key in ("estimated_cost_usd", "cost_per_1k"):
        val = row.get(key)
        if val is not None:
            return float(val or 0.0)
    return 0.0


def _is_acceptable(row: Dict[str, Any], quality_threshold: float) -> bool:
    if bool(row.get("abstained")):
        return False
    quality = row.get("quality")
    if quality is None:
        return True
    return float(quality) >= quality_threshold


def _load_query_rows(*, tenant_id: Optional[str], days: int, limit: int = 50_000) -> List[Dict[str, Any]]:
    ensure_query_log()
    since = datetime.utcnow() - timedelta(days=max(1, min(int(days), 365)))
    sql = """
        SELECT id, query_text, answer, chosen_model, modality, quality, abstained,
               latency_s, estimated_cost_usd, cost_per_1k, tenant_id, created_at
        FROM query_log
        WHERE created_at >= :since
    """
    params: Dict[str, Any] = {"since": since, "limit": limit}
    if tenant_id:
        sql += " AND tenant_id = :tenant_id"
        params["tenant_id"] = tenant_id
    sql += " ORDER BY id DESC LIMIT :limit"
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]
    except Exception:
        # Backward compatible when tenant_id column is not migrated yet
        if not tenant_id:
            return []
        sql_fallback = """
            SELECT id, query_text, answer, chosen_model, modality, quality, abstained,
                   latency_s, estimated_cost_usd, cost_per_1k, created_at
            FROM query_log
            WHERE created_at >= :since
            ORDER BY id DESC
            LIMIT :limit
        """
        with get_engine().connect() as conn:
            rows = conn.execute(text(sql_fallback), {"since": since, "limit": limit}).mappings().all()
        return [dict(r) for r in rows]


def build_roi_report(
    *,
    tenant_id: Optional[str] = None,
    days: int = 30,
    baseline_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Build savings report comparing actual router spend to baseline model."""
    baseline = _baseline_model_name(baseline_model)
    quality_threshold = _quality_threshold()
    rows = _load_query_rows(tenant_id=tenant_id, days=days)

    if not rows:
        usage = get_usage_summary(tenant_id) if tenant_id else get_usage_summary(None)
        return {
            "tenant_id": tenant_id,
            "period_days": days,
            "baseline_model": baseline,
            "quality_threshold": quality_threshold,
            "insufficient_data": True,
            "summary": {
                "query_count": 0,
                "actual_cost_usd": 0.0,
                "baseline_cost_usd": 0.0,
                "savings_usd": 0.0,
                "savings_pct": 0.0,
            },
            "usage_fallback": usage,
            "methodology": _methodology_block(baseline, quality_threshold),
        }

    actual_total = 0.0
    baseline_total = 0.0
    acceptable_actual = 0.0
    acceptable_baseline = 0.0
    acceptable_count = 0
    by_day: Dict[str, Dict[str, float]] = {}
    by_model: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        actual = _row_actual_cost(row)
        baseline_cost = _baseline_unit_cost(
            query_text=str(row.get("query_text") or ""),
            answer=str(row.get("answer") or ""),
            modality=str(row.get("modality") or "text"),
            baseline_model=baseline,
        )
        actual_total += actual
        baseline_total += baseline_cost

        acceptable = _is_acceptable(row, quality_threshold)
        if acceptable:
            acceptable_count += 1
            acceptable_actual += actual
            acceptable_baseline += baseline_cost

        day_key = str(row.get("created_at") or "")[:10] or "unknown"
        slot = by_day.setdefault(day_key, {"actual": 0.0, "baseline": 0.0, "count": 0})
        slot["actual"] += actual
        slot["baseline"] += baseline_cost
        slot["count"] += 1

        model = str(row.get("chosen_model") or "unknown")
        mslot = by_model.setdefault(
            model,
            {"model": model, "count": 0, "actual_cost_usd": 0.0, "baseline_cost_usd": 0.0, "quality_sum": 0.0},
        )
        mslot["count"] += 1
        mslot["actual_cost_usd"] += actual
        mslot["baseline_cost_usd"] += baseline_cost
        mslot["quality_sum"] += float(row.get("quality") or 0.0)

    savings = baseline_total - actual_total
    savings_pct = (savings / baseline_total * 100.0) if baseline_total > 0 else 0.0

    daily_series = []
    for day_key in sorted(by_day.keys()):
        bucket = by_day[day_key]
        day_savings = bucket["baseline"] - bucket["actual"]
        daily_series.append(
            {
                "date": day_key,
                "queries": int(bucket["count"]),
                "actual_cost_usd": round(bucket["actual"], 6),
                "baseline_cost_usd": round(bucket["baseline"], 6),
                "savings_usd": round(day_savings, 6),
            }
        )

    model_breakdown = []
    for model, bucket in sorted(by_model.items(), key=lambda x: x[1]["count"], reverse=True):
        model_savings = bucket["baseline_cost_usd"] - bucket["actual_cost_usd"]
        model_breakdown.append(
            {
                "model": model,
                "count": bucket["count"],
                "actual_cost_usd": round(bucket["actual_cost_usd"], 6),
                "baseline_cost_usd": round(bucket["baseline_cost_usd"], 6),
                "savings_usd": round(model_savings, 6),
                "quality_mean": round(bucket["quality_sum"] / max(1, bucket["count"]), 3),
            }
        )

    return {
        "tenant_id": tenant_id,
        "period_days": days,
        "baseline_model": baseline,
        "quality_threshold": quality_threshold,
        "insufficient_data": False,
        "summary": {
            "query_count": len(rows),
            "actual_cost_usd": round(actual_total, 6),
            "baseline_cost_usd": round(baseline_total, 6),
            "savings_usd": round(savings, 6),
            "savings_pct": round(savings_pct, 2),
            "acceptable_queries": acceptable_count,
            "cost_per_acceptable_actual_usd": round(acceptable_actual / acceptable_count, 6) if acceptable_count else None,
            "cost_per_acceptable_baseline_usd": round(acceptable_baseline / acceptable_count, 6) if acceptable_count else None,
            "projected_monthly_savings_usd": round(savings * (30 / max(1, days)), 2),
        },
        "daily_series": daily_series,
        "model_breakdown": model_breakdown[:15],
        "methodology": _methodology_block(baseline, quality_threshold),
        "disclaimer": (
            "Economia estimada vs baseline fixo. Não garante redução da fatura total se o volume de uso crescer."
        ),
    }


def _methodology_block(baseline_model: str, quality_threshold: float) -> Dict[str, Any]:
    return {
        "baseline_model": baseline_model,
        "baseline_description": "Contrafactual: todas as consultas no modelo premium de referência",
        "token_estimation": "chars / 4 (input + output)",
        "quality_threshold": quality_threshold,
        "acceptable_definition": f"quality >= {quality_threshold} e não abstido",
        "vision_surcharge_usd": VISION_BASELINE_SURCHARGE_USD,
        "env_overrides": ["ROI_BASELINE_MODEL", "ROI_QUALITY_THRESHOLD"],
    }
