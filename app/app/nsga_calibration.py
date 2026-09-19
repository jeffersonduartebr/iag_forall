# -*- coding: utf-8 -*-
# Objective: Periodic calibration cycle of the NSGA-II worker (risk, UQ, cache, predictor, judges).
"""Calibration steps run every few NSGA-II cycles, plus the status payload.

Each step is isolated: a failure is logged and the next step still runs.
Extracted from ``app.nsga_weights_updater`` (re-exported there).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.config.constants import DEFAULT_UNCERTAINTY_THRESHOLD
from app.services.nsga_tuning import calibrate_uncertainty_threshold, tune_risk_factors
from app.settings_dynamic import settings

logger = logging.getLogger("nsga-updater")


def _calibrate_risk_factors() -> None:
    result = tune_risk_factors()
    logger.info(f"[Calibration] Risk factors: {result.get('status', 'unknown')}")


def _calibrate_uncertainty() -> None:
    result = calibrate_uncertainty_threshold()
    logger.info(f"[Calibration] UQ threshold: {result.get('status', 'unknown')}")


def _publish_predictor_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    try:
        from app.observability import PREDICTOR_ACCURACY, PREDICTOR_BRIER_SCORE, PREDICTOR_CALIBRATION_TEMP

        for model, m in metrics.items():
            PREDICTOR_BRIER_SCORE.labels(model=model).set(m.get("brier_score", 0.25))
            PREDICTOR_ACCURACY.labels(model=model).set(m.get("accuracy", 0.5))
            PREDICTOR_CALIBRATION_TEMP.labels(model=model).set(m.get("calibration_temp", 1.0))
    except Exception:
        pass


def _calibrate_predictors() -> None:
    from app.online_predictor import calibrate_all_predictors, get_all_predictor_metrics

    calibrate_all_predictors()
    metrics = get_all_predictor_metrics()
    for model, m in metrics.items():
        logger.debug(f"[Calibration] Predictor {model}: brier={m.get('brier_score', 0):.3f}")
    _publish_predictor_metrics(metrics)


def _calibrate_judges() -> None:
    # Só banco de dados: app.judges puxaria providers/SDKs, ausentes na imagem enxuta do NSGA
    # ("No module named 'pybreaker'").
    from app.services.judge_calibration import calibrate_judges

    result = calibrate_judges()
    logger.info(f"[Calibration] Judge calibration: {result.get('status', 'unknown')}")


CALIBRATION_STEPS = (
    ("Risk factor tuning", _calibrate_risk_factors),
    ("UQ calibration", _calibrate_uncertainty),
    # O ajuste do limiar do cache semântico roda na API (router_core._cache_threshold_tuner):
    # ele lê a taxa de acerto do cache L1 do processo que atende as consultas.
    ("Predictor calibration", _calibrate_predictors),
    ("Judge calibration", _calibrate_judges),
)


def run_calibration_cycle() -> None:
    """Run every calibration step; one failing step never blocks the others."""
    logger.info("[Calibration] Starting calibration cycle...")
    for label, step in CALIBRATION_STEPS:
        try:
            step()
        except Exception as e:
            logger.warning(f"[Calibration] {label} failed: {e}")
    logger.info("[Calibration] Calibration cycle complete.")


def _optional(result: Dict[str, Any], section: str, key: str, loader) -> None:
    try:
        result[section][key] = loader()
    except Exception:
        pass


def calibration_status_payload() -> Dict[str, Any]:
    """Current calibration settings plus predictor, cache and judge metrics when available."""
    result: Dict[str, Any] = {
        "risk_factors": {
            "sota_high_uq": settings.RISK_FACTOR_SOTA_HIGH_UQ,
            "local_high_uq": settings.RISK_FACTOR_LOCAL_HIGH_UQ,
            "local_low_uq": settings.RISK_FACTOR_LOCAL_LOW_UQ,
            "adapt_enabled": settings.RISK_FACTOR_ADAPT_ENABLED,
        },
        "uncertainty": {
            "threshold": float(settings.get("UNCERTAINTY_THRESHOLD", DEFAULT_UNCERTAINTY_THRESHOLD)),
            "calibration_enabled": settings.UQ_CALIBRATION_ENABLED,
        },
        "cache": {
            "threshold": float(settings.get("CACHE_THRESHOLD", 0.92)),
            "adapt_enabled": settings.CACHE_THRESHOLD_ADAPT_ENABLED,
            "min": settings.CACHE_THRESHOLD_MIN,
            "max": settings.CACHE_THRESHOLD_MAX,
            "target_hit_rate": settings.CACHE_HIT_RATE_TARGET,
        },
        "predictor": {"validation_enabled": settings.PREDICTOR_VALIDATION_ENABLED},
        "judge": {
            "calibration_enabled": settings.JUDGE_CALIBRATION_ENABLED,
            "cache_agreement_target": settings.JUDGE_CACHE_AGREEMENT_TARGET,
        },
    }

    def _predictors():
        from app.online_predictor import get_all_predictor_metrics

        return get_all_predictor_metrics()

    def _l1_stats():
        from app.semantic_cache import get_l1_cache_stats

        return get_l1_cache_stats()

    def _hit_rate():
        from app.semantic_cache import get_cache_hit_rate

        return get_cache_hit_rate()

    def _judges():
        from app.judges import get_judge_calibration_metrics

        return get_judge_calibration_metrics()

    _optional(result, "predictor", "models", _predictors)
    _optional(result, "cache", "l1_stats", _l1_stats)
    _optional(result, "cache", "hit_rate", _hit_rate)
    _optional(result, "judge", "models", _judges)
    return result
