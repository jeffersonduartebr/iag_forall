# Objective: NSGA-II dynamic tuning extracted from nsga_weights_updater (roadmap #19).
"""Dynamic tuning routines: UQ threshold, global strategy weights, adaptive risk
factors, and UQ-threshold calibration. Extracted from nsga_weights_updater."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from app.config.constants import DEFAULT_UNCERTAINTY_THRESHOLD
from app.db import get_engine
from app.settings_dynamic import settings

from .frozen_policy import is_frozen_policy_active
from .nsga_metrics import NSGA_UQ_THRESH

logger = logging.getLogger("nsga-updater")


def _db_engine():
    return get_engine()


# ============================================================
# 5. Ajuste Dinâmico de Incerteza (UQ Tuning)
# ============================================================
def tune_uncertainty_threshold(current_efficiency: float) -> float:
    """Execute the tune uncertainty threshold routine.

    This helper encapsulates one focused step used by the surrounding workflow.
    Under frozen policy the threshold is left untouched so eval runs stay reproducible."""
    current_thresh = float(settings.get("UNCERTAINTY_THRESHOLD", DEFAULT_UNCERTAINTY_THRESHOLD))
    if is_frozen_policy_active():
        return current_thresh

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
_SOTA_MARKERS = ("gpt-5", "opus", "sonnet", "gemini-3-pro")
_MIN_BUCKET_SAMPLES = 20


@dataclass(frozen=True)
class RiskRule:
    """P-controller rule for one routing risk factor."""

    setting: str
    bucket: str
    label: str
    lower: float
    upper: float
    decrease_below: Optional[float]  # média de qualidade abaixo disso reduz o fator
    increase_above: Optional[float]  # média acima disso aumenta o fator


RISK_RULES = (
    # SOTA em alta incerteza: reduz o bônus se ficar abaixo de 7, aumenta se passar de 8.
    RiskRule("RISK_FACTOR_SOTA_HIGH_UQ", "sota_high_uq", "SOTA_HIGH", 1.0, 2.0, 7.0, 8.0),
    # Local em alta incerteza: penaliza mais abaixo de 5, alivia acima de 7.
    RiskRule("RISK_FACTOR_LOCAL_HIGH_UQ", "local_high_uq", "LOCAL_HIGH", 0.3, 1.0, 5.0, 7.0),
    # Local em terreno conhecido: só aumenta, acima de 7,5.
    RiskRule("RISK_FACTOR_LOCAL_LOW_UQ", "local_low_uq", "LOCAL_LOW", 0.0, 1.5, None, 7.5),
)


def bucket_qualities(rows: List[Any], uq_threshold: float) -> Dict[str, List[float]]:
    """Group judged qualities by (model family, uncertainty regime)."""
    buckets: Dict[str, List[float]] = {rule.bucket: [] for rule in RISK_RULES}
    for model, quality, uq in rows:
        lowered = str(model).lower()
        quality = float(quality) if quality else 5.0
        high_uq = (float(uq) if uq else 0.5) > uq_threshold
        if high_uq and any(marker in lowered for marker in _SOTA_MARKERS):
            buckets["sota_high_uq"].append(quality)
        elif "ollama" in lowered:
            buckets["local_high_uq" if high_uq else "local_low_uq"].append(quality)
    return buckets


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 5.0


def propose_risk_factor(rule: RiskRule, current: float, qualities: List[float], rate: float) -> Optional[float]:
    """New factor for ``rule`` (``None`` = keep): needs enough samples, moves by ``rate`` within bounds."""
    if len(qualities) < _MIN_BUCKET_SAMPLES:
        return None
    avg = _mean(qualities)
    if rule.decrease_below is not None and avg < rule.decrease_below:
        proposed = max(rule.lower, current - rate)
    elif rule.increase_above is not None and avg > rule.increase_above:
        proposed = min(rule.upper, current + rate)
    else:
        return None
    return proposed if proposed != current else None


def _bucket_metrics(buckets: Dict[str, List[float]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for name, values in buckets.items():
        metrics[f"{name}_count"] = len(values)
        metrics[f"{name}_avg_quality"] = _mean(values)
    return metrics


def _recent_quality_rows() -> List[Any]:
    with _db_engine().connect() as conn:
        return conn.execute(
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


def _publish_risk_factors() -> None:
    try:
        from app.observability import RISK_FACTOR_CURRENT

        for rule in RISK_RULES:
            RISK_FACTOR_CURRENT.labels(factor_type=rule.bucket).set(getattr(settings, rule.setting))
    except Exception:
        pass


def tune_risk_factors() -> Dict[str, Any]:
    """
    Tune risk factors based on observed quality outcomes by model type and UQ level.

    Analyzes the last 24 h of query_log: each rule in ``RISK_RULES`` nudges one
    factor by ``RISK_FACTOR_ADAPT_RATE`` (P-controller) when its bucket has at
    least 20 samples and the average quality crosses the rule's thresholds.

    Returns:
        Dict with adjustments made and metrics
    """
    if not settings.RISK_FACTOR_ADAPT_ENABLED:
        return {"status": "disabled"}
    try:
        rows = _recent_quality_rows()
        if len(rows) < 100:
            return {"status": "insufficient_data", "count": len(rows)}
        uq_threshold = float(settings.get("UNCERTAINTY_THRESHOLD", DEFAULT_UNCERTAINTY_THRESHOLD))
        buckets = bucket_qualities(rows, uq_threshold)
        adjustments: List[str] = []
        for rule in RISK_RULES:
            current = getattr(settings, rule.setting)
            proposed = propose_risk_factor(rule, current, buckets[rule.bucket], settings.RISK_FACTOR_ADAPT_RATE)
            if proposed is not None:
                settings.set(rule.setting, str(round(proposed, 2)), actor="risk-tuner")
                adjustments.append(f"{rule.label}: {current:.2f} -> {proposed:.2f}")
        _publish_risk_factors()
        if adjustments:
            logger.info(f"[Risk-Tuning] Adjustments: {', '.join(adjustments)}")
        return {"adjustments": adjustments, "metrics": _bucket_metrics(buckets), "status": "ok"}
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
    if is_frozen_policy_active():
        return {"status": "frozen"}

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

        current_threshold = float(settings.get("UNCERTAINTY_THRESHOLD", DEFAULT_UNCERTAINTY_THRESHOLD))
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
