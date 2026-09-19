# -*- coding: utf-8 -*-
# Objective: Judge calibration against cache agreement (extracted from judges.py).
"""Track judge scores against later cache outcomes and report calibration.

``record_judge_calibration`` stores each judgment; ``update_calibration_cache_status``
(called by the semantic cache on store) marks judgments whose answer got
cached; ``get_judge_calibration_metrics`` / ``calibrate_judges`` compare the two
("cache agreement") per judge model. Re-exported by ``app.judges``.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict

from sqlalchemy import text

from ..db import get_engine
from ..settings_dynamic import settings

logger = logging.getLogger("app.judges")


def _get_engine():
    """Shared database engine for calibration persistence."""
    return get_engine()


# Judge calibration table DDL
JUDGE_CALIBRATION_DDL = """
CREATE TABLE IF NOT EXISTS judge_calibration (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    judge_model VARCHAR(255) NOT NULL,
    query_hash VARCHAR(64) NOT NULL,
    predicted_score FLOAT NOT NULL,
    was_cached BOOLEAN DEFAULT FALSE,
    cache_hit_count INT DEFAULT 0,
    calibration_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_judge_model (judge_model),
    INDEX idx_query_hash (query_hash),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""


_table_ready = False


def _ensure_judge_calibration_table():
    """Ensure judge_calibration table exists (DDL runs once per process, not per judgment)."""
    global _table_ready
    if _table_ready:
        return
    try:
        with _get_engine().begin() as conn:
            conn.execute(text(JUDGE_CALIBRATION_DDL))
        _table_ready = True
    except Exception as exc:
        logger.warning("[Judges] Failed to create calibration table: %s", exc)



# ============================================================
# 🎯 Judge Calibration System (Phase 5 - Improvement 5)
# ============================================================

def record_judge_calibration(
    judge_model: str,
    query: str,
    predicted_score: float,
    was_cached: bool = False,
) -> None:
    """
    Record a judge's prediction for calibration analysis.

    Args:
        judge_model: The model that made the judgment
        query: The query text (will be hashed)
        predicted_score: The score assigned (0-10)
        was_cached: Whether this response was subsequently cached
    """
    if not settings.JUDGE_CALIBRATION_ENABLED:
        return

    try:
        _ensure_judge_calibration_table()
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:64]

        with _get_engine().begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO judge_calibration
                    (judge_model, query_hash, predicted_score, was_cached)
                    VALUES (:jm, :qh, :ps, :wc)
                """),
                {
                    "jm": judge_model,
                    "qh": query_hash,
                    "ps": predicted_score,
                    "wc": was_cached,
                },
            )
    except Exception as exc:
        logger.warning("[Judges] Calibration record fail: %s", exc)


def update_calibration_cache_status(query: str) -> None:
    """
    Update calibration records when a response is cached.

    Called from semantic_cache.store_cache() to track cache agreement.
    """
    if not settings.JUDGE_CALIBRATION_ENABLED:
        return

    try:
        # O cache pode gravar antes do primeiro julgamento: sem isto o UPDATE falhava com
        # "Table 'judge_calibration' doesn't exist".
        _ensure_judge_calibration_table()
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:64]

        with _get_engine().begin() as conn:
            # Update recent calibration records for this query
            conn.execute(
                text("""
                    UPDATE judge_calibration
                    SET was_cached = TRUE, cache_hit_count = cache_hit_count + 1
                    WHERE query_hash = :qh
                    AND created_at > NOW() - INTERVAL 1 HOUR
                """),
                {"qh": query_hash},
            )
    except Exception as exc:
        logger.warning("[Judges] Calibration update fail: %s", exc)


def get_judge_calibration_metrics() -> Dict[str, Dict[str, float]]:
    """
    Get calibration metrics for all judges.

    Returns:
        Dict mapping judge_model to metrics (cache_agreement, avg_score, etc.)
    """
    try:
        with _get_engine().connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        judge_model,
                        COUNT(*) as total_judgments,
                        AVG(predicted_score) as avg_score,
                        SUM(CASE WHEN was_cached THEN 1 ELSE 0 END) as cached_count,
                        SUM(CASE WHEN predicted_score >= 7.0 THEN 1 ELSE 0 END) as high_score_count,
                        SUM(CASE WHEN predicted_score >= 7.0 AND was_cached THEN 1 ELSE 0 END) as high_score_cached
                    FROM judge_calibration
                    WHERE created_at > NOW() - INTERVAL 24 HOUR
                    GROUP BY judge_model
                """)
            ).fetchall()

        result = {}
        for row in rows:
            judge_model = row[0]
            total = int(row[1]) if row[1] else 0
            avg_score = float(row[2]) if row[2] else 5.0
            cached_count = int(row[3]) if row[3] else 0
            high_score_count = int(row[4]) if row[4] else 0
            high_score_cached = int(row[5]) if row[5] else 0

            # Cache agreement: when judge gives high score, does it get cached?
            cache_agreement = high_score_cached / high_score_count if high_score_count > 0 else 0.0

            # Calibration score: correlation between high scores and caching
            calibration_score = cache_agreement  # Simple approximation

            result[judge_model] = {
                "total_judgments": total,
                "avg_score": avg_score,
                "cached_rate": cached_count / total if total > 0 else 0.0,
                "cache_agreement": cache_agreement,
                "calibration_score": calibration_score,
            }

        # Update Prometheus metrics
        try:
            from ..observability import JUDGE_CACHE_AGREEMENT, JUDGE_CALIBRATION_SCORE
            for model, metrics in result.items():
                JUDGE_CALIBRATION_SCORE.labels(judge_model=model).set(metrics["calibration_score"])
                JUDGE_CACHE_AGREEMENT.labels(judge_model=model).set(metrics["cache_agreement"])
        except Exception:
            pass

        return result

    except Exception as exc:
        logger.warning("[Judges] Failed to get calibration metrics: %s", exc)
        return {}


def calibrate_judges() -> Dict[str, Any]:
    """
    Analyze judge calibration and log insights.

    Called periodically to assess judge performance and alignment.

    Returns:
        Dict with calibration analysis results
    """
    if not settings.JUDGE_CALIBRATION_ENABLED:
        return {"status": "disabled"}

    metrics = get_judge_calibration_metrics()

    if not metrics:
        return {"status": "no_data"}

    target_agreement = settings.JUDGE_CACHE_AGREEMENT_TARGET
    warnings = []

    for model, data in metrics.items():
        agreement = data.get("cache_agreement", 0.0)
        if agreement < target_agreement and data.get("total_judgments", 0) > 50:
            warnings.append(
                f"{model}: cache_agreement={agreement:.2%} < target={target_agreement:.2%}"
            )

    if warnings:
        logger.warning(f"[Judge-Calibration] Low agreement: {', '.join(warnings)}")

    # Update metrics counter
    try:
        from ..observability import JUDGE_CALIBRATION_UPDATES
        JUDGE_CALIBRATION_UPDATES.inc()
    except Exception:
        pass

    return {
        "status": "ok",
        "metrics": metrics,
        "warnings": warnings,
    }
