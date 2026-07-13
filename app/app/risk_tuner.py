# -*- coding: utf-8 -*-
# Objective: Application runtime code for risk tuner.
"""
risk_tuner.py — Adaptive Risk Factor Management
------------------------------------------------
Dynamically adjusts risk factors based on observed model performance
across different uncertainty regimes.

Features:
- Tracks model performance by uncertainty level (high/low)
- Adjusts risk factors based on observed quality outcomes
- Uses rate-limited updates to prevent oscillation
- Enforces bounds on risk factors
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .observability import RISK_FACTOR_CURRENT
from .settings_dynamic import settings
from .utils.redis_client import get_redis

logger = logging.getLogger(__name__)

# Redis keys for persistence
REDIS_KEY_RISK_HISTORY = "risk_tuner:history"
REDIS_KEY_LAST_UPDATE = "risk_tuner:last_update"

# Bounds for risk factors
RISK_FACTOR_MIN = 0.3
RISK_FACTOR_MAX = 2.0


@dataclass
class PerformanceRecord:
    """Records model performance for a specific uncertainty regime."""
    total_samples: int = 0
    quality_sum: float = 0.0
    success_count: int = 0  # quality >= 7.0

    @property
    def avg_quality(self) -> float:
        """Execute the avg quality routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        if self.total_samples == 0:
            return 5.0
        return self.quality_sum / self.total_samples

    @property
    def success_rate(self) -> float:
        """Execute the success rate routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        if self.total_samples == 0:
            return 0.5
        return self.success_count / self.total_samples

    def to_dict(self) -> Dict[str, Any]:
        """Execute the to dict routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return {
            "total_samples": self.total_samples,
            "quality_sum": self.quality_sum,
            "success_count": self.success_count,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PerformanceRecord":
        """Execute the from dict routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return PerformanceRecord(
            total_samples=data.get("total_samples", 0),
            quality_sum=data.get("quality_sum", 0.0),
            success_count=data.get("success_count", 0),
        )


class AdaptiveRiskTuner:
    """
    Manages adaptive risk factor adjustments based on model performance.

    Tracks performance across uncertainty regimes and adjusts factors
    to optimize quality while respecting the current uncertainty level.
    """

    _instance: Optional["AdaptiveRiskTuner"] = None
    _lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> "AdaptiveRiskTuner":
        """Return the instance created for this class.

This hook is typically used to enforce singleton-style behavior or other allocation constraints."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
        if self._initialized:
            return

        # Performance tracking by (model_type, uncertainty_regime)
        # model_type: "sota" or "local"
        # uncertainty_regime: "high" or "low"
        self._performance: Dict[str, PerformanceRecord] = {
            "sota_high": PerformanceRecord(),
            "sota_low": PerformanceRecord(),
            "local_high": PerformanceRecord(),
            "local_low": PerformanceRecord(),
        }

        self._last_tune_time: float = 0.0
        self._min_samples_for_tune = 20  # Minimum samples before adjusting
        self._tune_cooldown_seconds = 300  # 5 minutes between adjustments

        # Load persisted state
        self._load_state()
        self._initialized = True

    def _get_redis(self):
        """Execute the get redis routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return get_redis()

    def _load_state(self) -> None:
        """Load persisted performance history from Redis."""
        rds = self._get_redis()
        if not rds:
            return

        try:
            raw = rds.get(REDIS_KEY_RISK_HISTORY)
            if raw:
                data = json.loads(raw)
                for key, record_data in data.items():
                    if key in self._performance:
                        self._performance[key] = PerformanceRecord.from_dict(record_data)
                logger.info("[RiskTuner] Loaded persisted state")

            last_update = rds.get(REDIS_KEY_LAST_UPDATE)
            if last_update:
                self._last_tune_time = float(last_update)
        except Exception as e:
            logger.warning(f"[RiskTuner] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Persist performance history to Redis."""
        rds = self._get_redis()
        if not rds:
            return

        try:
            data = {k: v.to_dict() for k, v in self._performance.items()}
            rds.set(REDIS_KEY_RISK_HISTORY, json.dumps(data))
            rds.set(REDIS_KEY_LAST_UPDATE, str(self._last_tune_time))
        except Exception as e:
            logger.warning(f"[RiskTuner] Failed to save state: {e}")

    def _get_model_type(self, model: str) -> str:
        """Determine if model is SOTA or local."""
        sota_markers = ["gpt-5", "opus", "sonnet", "gemini-3-pro", "gpt-4"]
        if any(m in model.lower() for m in sota_markers):
            return "sota"
        return "local"

    def record_outcome(
        self,
        model: str,
        quality: float,
        is_high_uncertainty: bool,
    ) -> None:
        """
        Record the outcome of a model call for risk tuning.

        Args:
            model: Model name
            quality: Quality score (0-10)
            is_high_uncertainty: Whether this was a high-uncertainty query
        """
        if not settings.RISK_FACTOR_ADAPT_ENABLED:
            return

        model_type = self._get_model_type(model)
        regime = "high" if is_high_uncertainty else "low"
        key = f"{model_type}_{regime}"

        if key not in self._performance:
            return

        record = self._performance[key]
        record.total_samples += 1
        record.quality_sum += quality
        if quality >= 7.0:
            record.success_count += 1

        # Periodically save state
        if record.total_samples % 10 == 0:
            self._save_state()

    def _calculate_adjustment(self, record: PerformanceRecord, current_factor: float) -> float:
        """
        Calculate the adjustment for a risk factor based on performance.

        Returns the new factor value.
        """
        if record.total_samples < self._min_samples_for_tune:
            return current_factor

        adapt_rate = settings.RISK_FACTOR_ADAPT_RATE
        success_rate = record.success_rate
        avg_quality = record.avg_quality

        # Target: success_rate > 0.7 and avg_quality > 7.0
        target_success = 0.7
        target_quality = 7.0

        # Calculate error terms
        success_error = target_success - success_rate
        quality_error = (target_quality - avg_quality) / 10.0  # Normalize to 0-1

        # Combined error (weighted average)
        combined_error = 0.6 * success_error + 0.4 * quality_error

        # Apply adjustment (positive error = need higher factor, negative = lower)
        adjustment = combined_error * adapt_rate
        new_factor = current_factor + adjustment

        # Clamp to bounds
        new_factor = max(RISK_FACTOR_MIN, min(RISK_FACTOR_MAX, new_factor))

        return round(new_factor, 3)

    def tune_factors(self) -> Dict[str, Any]:
        """
        Tune risk factors based on accumulated performance data.

        Returns dict with any adjustments made.
        """
        if not settings.RISK_FACTOR_ADAPT_ENABLED:
            return {"enabled": False}

        now = time.time()
        if now - self._last_tune_time < self._tune_cooldown_seconds:
            return {"cooldown": True, "seconds_remaining": self._tune_cooldown_seconds - (now - self._last_tune_time)}

        adjustments = {}

        # Tune SOTA high uncertainty factor
        sota_high_record = self._performance["sota_high"]
        current_sota_high = settings.RISK_FACTOR_SOTA_HIGH_UQ
        new_sota_high = self._calculate_adjustment(sota_high_record, current_sota_high)

        if new_sota_high != current_sota_high:
            settings.set("RISK_FACTOR_SOTA_HIGH_UQ", str(new_sota_high), actor="risk-tuner")
            adjustments["RISK_FACTOR_SOTA_HIGH_UQ"] = {"old": current_sota_high, "new": new_sota_high}
            RISK_FACTOR_CURRENT.labels(factor_type="sota_high_uq").set(new_sota_high)
            logger.info(f"[RiskTuner] Adjusted SOTA high UQ: {current_sota_high} -> {new_sota_high}")

        # Tune local high uncertainty factor
        local_high_record = self._performance["local_high"]
        current_local_high = settings.RISK_FACTOR_LOCAL_HIGH_UQ
        new_local_high = self._calculate_adjustment(local_high_record, current_local_high)

        if new_local_high != current_local_high:
            settings.set("RISK_FACTOR_LOCAL_HIGH_UQ", str(new_local_high), actor="risk-tuner")
            adjustments["RISK_FACTOR_LOCAL_HIGH_UQ"] = {"old": current_local_high, "new": new_local_high}
            RISK_FACTOR_CURRENT.labels(factor_type="local_high_uq").set(new_local_high)
            logger.info(f"[RiskTuner] Adjusted local high UQ: {current_local_high} -> {new_local_high}")

        # Tune local low uncertainty factor
        local_low_record = self._performance["local_low"]
        current_local_low = settings.RISK_FACTOR_LOCAL_LOW_UQ
        new_local_low = self._calculate_adjustment(local_low_record, current_local_low)

        if new_local_low != current_local_low:
            settings.set("RISK_FACTOR_LOCAL_LOW_UQ", str(new_local_low), actor="risk-tuner")
            adjustments["RISK_FACTOR_LOCAL_LOW_UQ"] = {"old": current_local_low, "new": new_local_low}
            RISK_FACTOR_CURRENT.labels(factor_type="local_low_uq").set(new_local_low)
            logger.info(f"[RiskTuner] Adjusted local low UQ: {current_local_low} -> {new_local_low}")

        self._last_tune_time = now
        self._save_state()

        return {
            "enabled": True,
            "adjustments": adjustments,
            "performance": {k: v.to_dict() for k, v in self._performance.items()},
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the risk tuner."""
        return {
            "enabled": settings.RISK_FACTOR_ADAPT_ENABLED,
            "current_factors": {
                "sota_high_uq": settings.RISK_FACTOR_SOTA_HIGH_UQ,
                "local_high_uq": settings.RISK_FACTOR_LOCAL_HIGH_UQ,
                "local_low_uq": settings.RISK_FACTOR_LOCAL_LOW_UQ,
            },
            "performance": {k: {
                "samples": v.total_samples,
                "avg_quality": round(v.avg_quality, 2),
                "success_rate": round(v.success_rate, 3),
            } for k, v in self._performance.items()},
            "last_tune_time": self._last_tune_time,
            "tune_cooldown": self._tune_cooldown_seconds,
            "min_samples": self._min_samples_for_tune,
        }

    def reset(self) -> None:
        """Reset all performance tracking data."""
        for key in self._performance:
            self._performance[key] = PerformanceRecord()
        self._last_tune_time = 0
        self._save_state()
        logger.info("[RiskTuner] Reset all performance data")


def get_risk_tuner() -> AdaptiveRiskTuner:
    """Get the global risk tuner instance."""
    return AdaptiveRiskTuner()
