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
_WEIGHT_LABELS = {"NSGA_W_LATENCY": "Lat", "NSGA_W_QUALITY": "Qual", "NSGA_W_COST": "Cost"}


def propose_strategy_weights(
    sys_metrics: Tuple[float, float, float], w_quality: float, w_latency: float, w_cost: float
) -> Dict[str, Tuple[float, float]]:
    """P-controller on the portfolio metrics: ``setting -> (old, new)`` for the weights that move.

    Latency target < 3 s (relaxed below 1 s), quality > 7, cost < 0.01 USD/request.
    """
    sys_lat, sys_cst, sys_qlt = sys_metrics
    proposals: Dict[str, Tuple[float, float]] = {}
    if sys_lat > 3.0:
        proposals["NSGA_W_LATENCY"] = (w_latency, min(2.0, w_latency + 0.1))
    elif sys_lat < 1.0:
        proposals["NSGA_W_LATENCY"] = (w_latency, max(0.1, w_latency - 0.05))
    if sys_qlt < 7.0:
        proposals["NSGA_W_QUALITY"] = (w_quality, min(5.0, w_quality + 0.2))
    if sys_cst > 0.01:
        proposals["NSGA_W_COST"] = (w_cost, min(100.0, w_cost + 5.0))
    return {key: (old, new) for key, (old, new) in proposals.items() if new != old}


def tune_global_strategy_weights(sys_metrics: Tuple[float, float, float]):
    """
    Ajusta os pesos globais (NSGA_W_QUALITY, etc.) baseado no desempenho
    do melhor indivíduo encontrado pelo AG.

    Se o sistema ideal encontrado ainda é lento, aumentamos a penalidade de latência.
    Se a qualidade está baixa, aumentamos o peso da qualidade.
    """
    if is_frozen_policy_active():  # NSGA_W_* fazem parte do snapshot congelado da avaliação
        return
    changes = propose_strategy_weights(sys_metrics, settings.NSGA_W_QUALITY, settings.NSGA_W_LATENCY, settings.NSGA_W_COST)
    for key, (_old, new) in changes.items():
        settings.set(key, str(round(new, 2)), actor="nsga-updater")
    if changes:
        applied = ", ".join(f"{_WEIGHT_LABELS[key]} {old}->{new:.2f}" for key, (old, new) in changes.items())
        logger.info(f"[Strategy-Tuning] Ajustes aplicados: {applied}")


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


def _or_default(value: Any, default: float) -> float:
    """Missing (NULL) values take the default; a real 0.0 (abstention, exact centroid match) is kept."""
    return default if value is None or value == "" else float(value)


def bucket_qualities(rows: List[Any], uq_threshold: float) -> Dict[str, List[float]]:
    """Group judged qualities by (model family, uncertainty regime)."""
    buckets: Dict[str, List[float]] = {rule.bucket: [] for rule in RISK_RULES}
    for model, quality, uq in rows:
        lowered = str(model).lower()
        quality = _or_default(quality, 5.0)
        high_uq = _or_default(uq, 0.5) > uq_threshold
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
def split_by_uncertainty(rows: List[Any], threshold: float) -> Tuple[List[float], List[float]]:
    """Qualities of ``(model, quality, uq)`` rows above / at-or-below the uncertainty threshold."""
    high: List[float] = []
    low: List[float] = []
    for _model, quality, uq in rows:
        (high if _or_default(uq, 0.5) > threshold else low).append(_or_default(quality, 5.0))
    return high, low


def decide_threshold(current: float, quality_gap: float, gap_relax: float, gap_tighten: float) -> Tuple[float, str]:
    """Relax when high-UQ answers are barely worse, tighten when they are much worse (bounds 0.20-0.80)."""
    if quality_gap < gap_relax:
        return min(0.80, current + 0.05), "RELAX"
    if quality_gap > gap_tighten:
        return max(0.20, current - 0.05), "TIGHTEN"
    return current, "KEEP"


def _publish_uq_metrics(avg_high: float, avg_low: float, quality_gap: float) -> None:
    try:
        from app.observability import UQ_HIGH_AVG_QUALITY, UQ_LOW_AVG_QUALITY, UQ_VS_ERROR_CORRELATION

        UQ_HIGH_AVG_QUALITY.set(avg_high)
        UQ_LOW_AVG_QUALITY.set(avg_low)
        # Aproximação da correlação: gap de qualidade normalizado
        UQ_VS_ERROR_CORRELATION.set(min(1.0, max(-1.0, quality_gap / 5.0)))
    except Exception:
        pass


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
    try:
        rows = _recent_quality_rows()
        if len(rows) < 100:
            return {"status": "insufficient_data", "count": len(rows)}
        current = float(settings.get("UNCERTAINTY_THRESHOLD", DEFAULT_UNCERTAINTY_THRESHOLD))
        high, low = split_by_uncertainty(rows, current)
        if len(high) < _MIN_BUCKET_SAMPLES or len(low) < _MIN_BUCKET_SAMPLES:
            return {"status": "insufficient_split", "high_count": len(high), "low_count": len(low)}

        avg_high, avg_low = _mean(high), _mean(low)
        quality_gap = avg_low - avg_high
        _publish_uq_metrics(avg_high, avg_low, quality_gap)
        new_threshold, action = decide_threshold(
            current, quality_gap, settings.UQ_QUALITY_GAP_RELAX, settings.UQ_QUALITY_GAP_TIGHTEN
        )
        if action != "KEEP":
            settings.set("UNCERTAINTY_THRESHOLD", str(round(new_threshold, 2)), actor="uq-calibrator")
            NSGA_UQ_THRESH.set(new_threshold)
            logger.info(
                f"[UQ-Calibration] {action}: threshold {current:.2f} -> {new_threshold:.2f} (gap={quality_gap:.2f})"
            )
        return {
            "old_threshold": current,
            "new_threshold": new_threshold,
            "action": action,
            "metrics": {
                "high_uq_count": len(high),
                "low_uq_count": len(low),
                "avg_quality_high_uq": avg_high,
                "avg_quality_low_uq": avg_low,
                "quality_gap": quality_gap,
            },
            "status": "ok",
        }
    except Exception as e:
        logger.warning(f"[UQ-Calibration] Failed: {e}")
        return {"status": "error", "error": str(e)}
