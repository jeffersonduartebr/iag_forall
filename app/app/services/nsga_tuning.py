# Objective: NSGA-II dynamic tuning extracted from nsga_weights_updater (roadmap #19).
"""Dynamic tuning routines: UQ threshold, global strategy weights, adaptive risk
factors, and UQ-threshold calibration. Extracted from nsga_weights_updater."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from sqlalchemy import text

from app.db import get_engine
from app.settings_dynamic import settings

from .nsga_metrics import NSGA_UQ_THRESH

logger = logging.getLogger("nsga-updater")


def _db_engine():
    return get_engine()


# ============================================================
# 5. Ajuste Dinâmico de Incerteza (UQ Tuning)
# ============================================================
def tune_uncertainty_threshold(current_efficiency: float) -> float:
    """Execute the tune uncertainty threshold routine.

    This helper encapsulates one focused step used by the surrounding workflow."""
    current_thresh = float(settings.get("UNCERTAINTY_THRESHOLD", 0.45))

    if current_efficiency < 2.0:
        new_thresh = max(0.20, current_thresh - 0.05)
        action = "TIGHTEN"
    elif current_efficiency > 4.0:
        new_thresh = min(0.80, current_thresh + 0.05)
        action = "RELAX"
    else:
        new_thresh = current_thresh
        action = "KEEP"

    if action != "KEEP":
        logger.info(
            f"[UQ-Tuning] Eficiência={current_efficiency:.2f}. {action} Threshold: {current_thresh:.2f} -> {new_thresh:.2f}"
        )
        settings.set("UNCERTAINTY_THRESHOLD", str(new_thresh), actor="nsga-updater")
        NSGA_UQ_THRESH.set(new_thresh)

    return new_thresh


# ============================================================
# 6. Ajuste Dinâmico de Pesos Globais (Strategy Tuning) - NOVO
# ============================================================
def tune_global_strategy_weights(sys_metrics: Tuple[float, float, float]):
    """
    Ajusta os pesos globais (NSGA_W_QUALITY, etc.) baseado no desempenho
    do melhor indivíduo encontrado pelo AG.

    Se o sistema ideal encontrado ainda é lento, aumentamos a penalidade de latência.
    Se a qualidade está baixa, aumentamos o peso da qualidade.
    """
    sys_lat, sys_cst, sys_qlt = sys_metrics

    # Lê valores atuais
    w_qual = settings.NSGA_W_QUALITY
    w_lat = settings.NSGA_W_LATENCY
    w_cost = settings.NSGA_W_COST

    changes = []

    # --- Lógica de Controle (P-Controller simples) ---

    # 1. Controle de Latência (Target: < 3.0s)
    if sys_lat > 3.0:
        # Sistema lento -> Aumenta importância da latência
        new_w_lat = min(2.0, w_lat + 0.1)
        if new_w_lat != w_lat:
            settings.set("NSGA_W_LATENCY", str(round(new_w_lat, 2)), actor="nsga-updater")
            changes.append(f"Lat {w_lat}->{new_w_lat:.2f}")
    elif sys_lat < 1.0:
        # Sistema muito rápido -> Relaxa latência para ganhar qualidade
        new_w_lat = max(0.1, w_lat - 0.05)
        if new_w_lat != w_lat:
            settings.set("NSGA_W_LATENCY", str(round(new_w_lat, 2)), actor="nsga-updater")
            changes.append(f"Lat {w_lat}->{new_w_lat:.2f}")

    # 2. Controle de Qualidade (Target: > 7.0)
    if sys_qlt < 7.0:
        # Qualidade baixa -> Aumenta importância da qualidade
        new_w_qual = min(5.0, w_qual + 0.2)
        if new_w_qual != w_qual:
            settings.set("NSGA_W_QUALITY", str(round(new_w_qual, 2)), actor="nsga-updater")
            changes.append(f"Qual {w_qual}->{new_w_qual:.2f}")

    # 3. Controle de Custo (Target: < $0.01/req)
    if sys_cst > 0.01:
        new_w_cost = min(100.0, w_cost + 5.0)
        if new_w_cost != w_cost:
            settings.set("NSGA_W_COST", str(round(new_w_cost, 2)), actor="nsga-updater")
            changes.append(f"Cost {w_cost}->{new_w_cost:.2f}")

    if changes:
        logger.info(f"[Strategy-Tuning] Ajustes aplicados: {', '.join(changes)}")


# ============================================================
# 6.1 Adaptive Risk Factors Tuning (Phase 5 - Improvement 1)
# ============================================================
def tune_risk_factors() -> Dict[str, Any]:
    """
    Tune risk factors based on observed quality outcomes by model type and UQ level.

    Analyzes query_log data to see if risk factor adjustments improve quality:
    - If SOTA models in high-UQ scenarios underperform, reduce their boost
    - If local models in high-UQ scenarios outperform expectations, increase their factor

    Uses P-controller feedback to make gradual adjustments.

    Returns:
        Dict with adjustments made and metrics
    """
    if not settings.RISK_FACTOR_ADAPT_ENABLED:
        return {"status": "disabled"}

    result: dict[str, Any] = {"adjustments": [], "metrics": {}}

    try:
        with _db_engine().connect() as conn:
            # Query recent data with UQ scores from raw_payload
            rows = conn.execute(
                text("""
                    SELECT
                        chosen_model,
                        quality,
                        JSON_EXTRACT(raw_payload, '$.uncertainty_score') as uq_score
                    FROM query_log
                    WHERE created_at > NOW() - INTERVAL 24 HOUR
                    AND quality IS NOT NULL
                    AND raw_payload IS NOT NULL
                    AND JSON_EXTRACT(raw_payload, '$.uncertainty_score') IS NOT NULL
                    LIMIT 5000
                """)
            ).fetchall()

        if len(rows) < 100:
            return {"status": "insufficient_data", "count": len(rows)}

        # Categorize results
        sota_high_uq = []
        local_high_uq = []
        local_low_uq = []
        uq_threshold = float(settings.get("UNCERTAINTY_THRESHOLD", 0.45))

        for row in rows:
            model = row[0]
            quality = float(row[1]) if row[1] else 5.0
            uq_score = float(row[2]) if row[2] else 0.5

            is_sota = any(m in model.lower() for m in ["gpt-5", "opus", "sonnet", "gemini-3-pro"])
            is_local = "ollama" in model.lower()
            is_high_uq = uq_score > uq_threshold

            if is_high_uq:
                if is_sota:
                    sota_high_uq.append(quality)
                elif is_local:
                    local_high_uq.append(quality)
            else:
                if is_local:
                    local_low_uq.append(quality)

        # Calculate average qualities
        def safe_mean(lst):
            """Executa a responsabilidade descrita por este método.

            Args:
                lst: Parâmetro de entrada.

            Returns:
                Valor produzido pela execução.
            """
            return sum(lst) / len(lst) if lst else 5.0

        avg_sota_high = safe_mean(sota_high_uq)
        avg_local_high = safe_mean(local_high_uq)
        avg_local_low = safe_mean(local_low_uq)

        result["metrics"] = {
            "sota_high_uq_count": len(sota_high_uq),
            "sota_high_uq_avg_quality": avg_sota_high,
            "local_high_uq_count": len(local_high_uq),
            "local_high_uq_avg_quality": avg_local_high,
            "local_low_uq_count": len(local_low_uq),
            "local_low_uq_avg_quality": avg_local_low,
        }

        adapt_rate = settings.RISK_FACTOR_ADAPT_RATE

        # Tune SOTA high-UQ factor
        current_sota = settings.RISK_FACTOR_SOTA_HIGH_UQ
        if len(sota_high_uq) >= 20:
            # If SOTA is underperforming in high-UQ (< 7.0), reduce boost
            if avg_sota_high < 7.0:
                new_sota = max(1.0, current_sota - adapt_rate)
                if new_sota != current_sota:
                    settings.set("RISK_FACTOR_SOTA_HIGH_UQ", str(round(new_sota, 2)), actor="risk-tuner")
                    result["adjustments"].append(f"SOTA_HIGH: {current_sota:.2f} -> {new_sota:.2f}")
            # If SOTA is performing well, slight boost
            elif avg_sota_high > 8.0:
                new_sota = min(2.0, current_sota + adapt_rate)
                if new_sota != current_sota:
                    settings.set("RISK_FACTOR_SOTA_HIGH_UQ", str(round(new_sota, 2)), actor="risk-tuner")
                    result["adjustments"].append(f"SOTA_HIGH: {current_sota:.2f} -> {new_sota:.2f}")

        # Tune local high-UQ factor
        current_local_high = settings.RISK_FACTOR_LOCAL_HIGH_UQ
        if len(local_high_uq) >= 20:
            # If local models are underperforming in high-UQ, reduce further
            if avg_local_high < 5.0:
                new_local = max(0.3, current_local_high - adapt_rate)
                if new_local != current_local_high:
                    settings.set("RISK_FACTOR_LOCAL_HIGH_UQ", str(round(new_local, 2)), actor="risk-tuner")
                    result["adjustments"].append(f"LOCAL_HIGH: {current_local_high:.2f} -> {new_local:.2f}")
            # If local models are surprisingly good, increase
            elif avg_local_high > 7.0:
                new_local = min(1.0, current_local_high + adapt_rate)
                if new_local != current_local_high:
                    settings.set("RISK_FACTOR_LOCAL_HIGH_UQ", str(round(new_local, 2)), actor="risk-tuner")
                    result["adjustments"].append(f"LOCAL_HIGH: {current_local_high:.2f} -> {new_local:.2f}")

        # Tune local low-UQ factor
        current_local_low = settings.RISK_FACTOR_LOCAL_LOW_UQ
        if len(local_low_uq) >= 20:
            # If local models do well in known territory, boost
            if avg_local_low > 7.5:
                new_local = min(1.5, current_local_low + adapt_rate)
                if new_local != current_local_low:
                    settings.set("RISK_FACTOR_LOCAL_LOW_UQ", str(round(new_local, 2)), actor="risk-tuner")
                    result["adjustments"].append(f"LOCAL_LOW: {current_local_low:.2f} -> {new_local:.2f}")

        # Update Prometheus metrics
        try:
            from app.observability import RISK_FACTOR_CURRENT

            RISK_FACTOR_CURRENT.labels(factor_type="sota_high_uq").set(settings.RISK_FACTOR_SOTA_HIGH_UQ)
            RISK_FACTOR_CURRENT.labels(factor_type="local_high_uq").set(settings.RISK_FACTOR_LOCAL_HIGH_UQ)
            RISK_FACTOR_CURRENT.labels(factor_type="local_low_uq").set(settings.RISK_FACTOR_LOCAL_LOW_UQ)
        except Exception:
            pass

        if result["adjustments"]:
            logger.info(f"[Risk-Tuning] Adjustments: {', '.join(result['adjustments'])}")

        result["status"] = "ok"
        return result

    except Exception as e:
        logger.warning(f"[Risk-Tuning] Failed: {e}")
        return {"status": "error", "error": str(e)}


# ============================================================
# 6.2 UQ Calibration Against Actual Errors (Phase 5 - Improvement 4)
# ============================================================
def calibrate_uncertainty_threshold() -> Dict[str, Any]:
    """
    Calibrate uncertainty threshold based on actual quality outcomes.

    Analyzes if high-UQ queries actually have lower quality than low-UQ queries.
    If the gap is small, the threshold can be relaxed.
    If the gap is large, the threshold should be tightened.

    Returns:
        Dict with calibration results and metrics
    """
    if not settings.UQ_CALIBRATION_ENABLED:
        return {"status": "disabled"}

    result: dict[str, Any] = {"old_threshold": None, "new_threshold": None, "metrics": {}}

    try:
        with _db_engine().connect() as conn:
            # Query data grouped by UQ level
            rows = conn.execute(
                text("""
                    SELECT
                        quality,
                        JSON_EXTRACT(raw_payload, '$.uncertainty_score') as uq_score
                    FROM query_log
                    WHERE created_at > NOW() - INTERVAL 24 HOUR
                    AND quality IS NOT NULL
                    AND raw_payload IS NOT NULL
                    AND JSON_EXTRACT(raw_payload, '$.uncertainty_score') IS NOT NULL
                    LIMIT 5000
                """)
            ).fetchall()

        if len(rows) < 100:
            return {"status": "insufficient_data", "count": len(rows)}

        current_threshold = float(settings.get("UNCERTAINTY_THRESHOLD", 0.45))
        result["old_threshold"] = current_threshold

        high_uq_qualities = []
        low_uq_qualities = []

        for row in rows:
            quality = float(row[0]) if row[0] else 5.0
            uq_score = float(row[1]) if row[1] else 0.5

            if uq_score > current_threshold:
                high_uq_qualities.append(quality)
            else:
                low_uq_qualities.append(quality)

        if len(high_uq_qualities) < 20 or len(low_uq_qualities) < 20:
            return {
                "status": "insufficient_split",
                "high_count": len(high_uq_qualities),
                "low_count": len(low_uq_qualities),
            }

        avg_quality_high = sum(high_uq_qualities) / len(high_uq_qualities)
        avg_quality_low = sum(low_uq_qualities) / len(low_uq_qualities)
        quality_gap = avg_quality_low - avg_quality_high

        result["metrics"] = {
            "high_uq_count": len(high_uq_qualities),
            "low_uq_count": len(low_uq_qualities),
            "avg_quality_high_uq": avg_quality_high,
            "avg_quality_low_uq": avg_quality_low,
            "quality_gap": quality_gap,
        }

        # Update Prometheus metrics
        try:
            from app.observability import UQ_HIGH_AVG_QUALITY, UQ_LOW_AVG_QUALITY, UQ_VS_ERROR_CORRELATION

            UQ_HIGH_AVG_QUALITY.set(avg_quality_high)
            UQ_LOW_AVG_QUALITY.set(avg_quality_low)
            # Correlation approximation: quality_gap normalized
            correlation_approx = min(1.0, max(-1.0, quality_gap / 5.0))
            UQ_VS_ERROR_CORRELATION.set(correlation_approx)
        except Exception:
            pass

        # Threshold adjustment logic
        gap_relax = settings.UQ_QUALITY_GAP_RELAX
        gap_tighten = settings.UQ_QUALITY_GAP_TIGHTEN

        if quality_gap < gap_relax:
            # High-UQ queries aren't much worse -> relax threshold
            new_threshold = min(0.80, current_threshold + 0.05)
            action = "RELAX"
        elif quality_gap > gap_tighten:
            # High-UQ queries are much worse -> tighten threshold
            new_threshold = max(0.20, current_threshold - 0.05)
            action = "TIGHTEN"
        else:
            new_threshold = current_threshold
            action = "KEEP"

        result["new_threshold"] = new_threshold
        result["action"] = action

        if action != "KEEP":
            settings.set("UNCERTAINTY_THRESHOLD", str(round(new_threshold, 2)), actor="uq-calibrator")
            NSGA_UQ_THRESH.set(new_threshold)
            logger.info(
                f"[UQ-Calibration] {action}: threshold {current_threshold:.2f} -> {new_threshold:.2f} "
                f"(gap={quality_gap:.2f})"
            )

        result["status"] = "ok"
        return result

    except Exception as e:
        logger.warning(f"[UQ-Calibration] Failed: {e}")
        return {"status": "error", "error": str(e)}
