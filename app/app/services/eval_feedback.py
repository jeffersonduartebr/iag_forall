# Objective: Apply evaluation run outcomes to NSGA weights and bandit exploration.
"""Bridge completed eval runs into online routing policy tuning."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

REDIS_EVAL_FEEDBACK_KEY = "eval:feedback:latest"


def _settings():
    from app.settings_dynamic import settings

    return settings


def _redis_client():
    try:
        from app.utils.redis_client import get_redis

        return get_redis()
    except Exception:
        return None


def apply_eval_run_feedback(
    run_id: str,
    summary: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Tune NSGA global weights and bandit epsilon from one completed eval run."""
    summary = dict(summary or {})
    metadata = dict(metadata or {})
    try:
        from app.services.frozen_policy import should_skip_eval_feedback

        if should_skip_eval_feedback(metadata):
            return {
                "run_id": run_id,
                "skipped": True,
                "reason": "frozen_policy",
            }
    except Exception:
        pass
    quality_mean = float(summary.get("quality_mean", 0.0) or 0.0)
    latency_mean = float(summary.get("latency_mean", 0.0) or 0.0)
    cost_mean = float(summary.get("cost_mean", 0.0) or 0.0)

    settings = _settings()

    changes: Dict[str, Any] = {
        "run_id": run_id,
        "quality_mean": quality_mean,
        "latency_mean": latency_mean,
        "cost_mean": cost_mean,
        "nsga_changes": [],
        "bandit_changes": [],
    }

    try:
        changes["nsga_changes"] = _apply_nsga_feedback(settings, latency_mean, cost_mean, quality_mean)
    except Exception as exc:
        logger.warning("[eval-feedback] NSGA tuning skipped: %s", exc)
        changes["nsga_error"] = str(exc)

    quality_min = float(settings.get("EVAL_FEEDBACK_QUALITY_MIN", 6.0))
    quality_max = float(settings.get("EVAL_FEEDBACK_QUALITY_MAX", 8.5))
    if quality_mean < quality_min:
        eps = float(settings.get("BANDIT_EPSILON", 0.1))
        new_eps = min(float(settings.get("EVAL_FEEDBACK_BANDIT_EPSILON_MAX", 0.35)), eps + 0.02)
        if new_eps != eps:
            settings.set("BANDIT_EPSILON", str(round(new_eps, 4)), actor="eval-feedback", source="eval_run")
            changes["bandit_changes"].append({"setting": "BANDIT_EPSILON", "before": eps, "after": new_eps})
    elif quality_mean > quality_max:
        eps = float(settings.get("BANDIT_EPSILON", 0.1))
        new_eps = max(float(settings.get("EVAL_FEEDBACK_BANDIT_EPSILON_MIN", 0.05)), eps - 0.01)
        if new_eps != eps:
            settings.set("BANDIT_EPSILON", str(round(new_eps, 4)), actor="eval-feedback", source="eval_run")
            changes["bandit_changes"].append({"setting": "BANDIT_EPSILON", "before": eps, "after": new_eps})

    theme_breakdown = _theme_quality_breakdown(run_id)
    changes["theme_breakdown"] = theme_breakdown

    _persist_feedback(changes)
    logger.info(
        "[eval-feedback] run=%s quality=%.2f nsga_changes=%s bandit_changes=%s",
        run_id,
        quality_mean,
        len(changes["nsga_changes"]),
        len(changes["bandit_changes"]),
    )
    return changes


def _apply_nsga_feedback(settings: Any, latency_mean: float, cost_mean: float, quality_mean: float) -> list[Dict[str, Any]]:
    from app.nsga_weights_updater import tune_global_strategy_weights

    before = {
        "NSGA_W_QUALITY": float(settings.NSGA_W_QUALITY),
        "NSGA_W_LATENCY": float(settings.NSGA_W_LATENCY),
        "NSGA_W_COST": float(settings.NSGA_W_COST),
    }
    tune_global_strategy_weights((latency_mean, cost_mean, quality_mean))
    after = {
        "NSGA_W_QUALITY": float(settings.NSGA_W_QUALITY),
        "NSGA_W_LATENCY": float(settings.NSGA_W_LATENCY),
        "NSGA_W_COST": float(settings.NSGA_W_COST),
    }
    nsga_changes: list[Dict[str, Any]] = []
    for key in before:
        if before[key] != after[key]:
            nsga_changes.append({"setting": key, "before": before[key], "after": after[key]})
    return nsga_changes


def _theme_quality_breakdown(run_id: str) -> Dict[str, float]:
    """Aggregate mean quality per benchmark theme from eval result metadata."""
    from app.roadmap_features import list_eval_run_results

    results = list_eval_run_results(run_id, limit=5000)
    buckets: Dict[str, list[float]] = {}
    for row in results:
        meta = row.get("metadata") or {}
        theme = str(meta.get("benchmark_theme") or "unknown")
        quality = float(row.get("quality", 0.0) or 0.0)
        buckets.setdefault(theme, []).append(quality)
    return {theme: (sum(vals) / len(vals)) for theme, vals in buckets.items() if vals}


def _persist_feedback(payload: Dict[str, Any]) -> None:
    rds = _redis_client()
    if not rds:
        return
    try:
        rds.set(REDIS_EVAL_FEEDBACK_KEY, json.dumps(payload, ensure_ascii=False))
        rds.set(f"eval:feedback:{payload.get('run_id')}", json.dumps(payload, ensure_ascii=False), ex=86400 * 14)
    except Exception as exc:
        logger.debug("[eval-feedback] redis persist skipped: %s", exc)


def get_latest_eval_feedback() -> Optional[Dict[str, Any]]:
    """Return the most recent eval feedback payload from Redis."""
    rds = _redis_client()
    if not rds:
        return None
    try:
        raw = rds.get(REDIS_EVAL_FEEDBACK_KEY)
        if not raw:
            return None
        text = raw.decode() if isinstance(raw, bytes) else str(raw)
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None
