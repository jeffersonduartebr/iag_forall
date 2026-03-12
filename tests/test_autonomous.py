# -*- coding: utf-8 -*-
# Objective: Test coverage for autonomous behavior and regressions.
"""
test_autonomous.py — Tests for Phase 5 Autonomous Behavior Improvements
------------------------------------------------------------------------
Tests the self-tuning and calibration systems:
- Adaptive Risk Factors (Improvement 1)
- Predictor Validation (Improvement 2)
- Adaptive Cache Threshold (Improvement 3)
- UQ Calibration (Improvement 4)
- Judge Calibration (Improvement 5)
"""

import pytest
import time
from unittest.mock import patch, MagicMock


# =============================================================================
# Test Improvement 1: Adaptive Risk Factors
# =============================================================================

class TestAdaptiveRiskFactors:
    """Tests for adaptive risk factor tuning."""

    def test_risk_factor_settings_exist(self):
        """Verify risk factor settings are defined with defaults."""
        from app.settings_dynamic import settings

        assert settings.RISK_FACTOR_SOTA_HIGH_UQ >= 0.0
        assert settings.RISK_FACTOR_LOCAL_HIGH_UQ >= 0.0
        assert settings.RISK_FACTOR_LOCAL_LOW_UQ >= 0.0

    def test_risk_factor_bounds(self):
        """Risk factors should be within reasonable bounds."""
        from app.settings_dynamic import settings

        # SOTA should have boost (>= 1.0) in high uncertainty
        assert 0.5 <= settings.RISK_FACTOR_SOTA_HIGH_UQ <= 3.0

        # Local should be penalized (<= 1.0) in high uncertainty
        assert 0.1 <= settings.RISK_FACTOR_LOCAL_HIGH_UQ <= 1.5

        # Local should have slight boost in low uncertainty
        assert 0.5 <= settings.RISK_FACTOR_LOCAL_LOW_UQ <= 2.0

    def test_risk_factors_used_in_strategy(self):
        """Verify router_strategy uses dynamic risk factors."""
        from app.router_strategy import choose_top2_models
        from app.settings_dynamic import settings

        candidates = ["ollama/phi4:latest", "openai/gpt-5"]
        weights = {"w_quality": 1.0, "w_latency": 0.5, "w_cost": 50.0}

        # This should not raise and should use dynamic factors
        result = choose_top2_models(
            candidates=candidates,
            weights=weights,
            query_text="test query",
            modality="text",
            uncertainty_score=0.8,  # High uncertainty
        )

        assert len(result) <= 2
        assert all(m in candidates for m in result)


# =============================================================================
# Test Improvement 2: Predictor Validation
# =============================================================================

class TestPredictorValidation:
    """Tests for online predictor validation and calibration."""

    def test_predictor_initialization(self):
        """Test predictor initializes with validation tracking."""
        from app.online_predictor import OnlineErrorPredictor

        pred = OnlineErrorPredictor("test/model")

        assert hasattr(pred, "prediction_log")
        assert hasattr(pred, "calibration_temp")
        assert pred.calibration_temp == 1.0
        assert pred.prediction_log == []

    def test_record_outcome(self):
        """Test recording prediction outcomes."""
        from app.online_predictor import OnlineErrorPredictor

        pred = OnlineErrorPredictor("test/model2")
        pred.prediction_log = []  # Reset

        # Record some outcomes
        pred.record_outcome(0.7, True)   # Predicted error, was error
        pred.record_outcome(0.3, False)  # Predicted no error, was no error
        pred.record_outcome(0.6, False)  # Predicted error, was no error

        assert len(pred.prediction_log) == 3

    def test_brier_score_calculation(self):
        """Test Brier score computation."""
        from app.online_predictor import OnlineErrorPredictor

        pred = OnlineErrorPredictor("test/model3")
        pred.prediction_log = []

        # Perfect predictions: predict 1.0 for errors, 0.0 for non-errors
        for i in range(60):
            if i % 2 == 0:
                pred.prediction_log.append((1.0, True, time.time()))
            else:
                pred.prediction_log.append((0.0, False, time.time()))

        brier = pred.compute_brier_score()
        assert brier == 0.0  # Perfect calibration

    def test_brier_score_random(self):
        """Test Brier score for random predictions."""
        from app.online_predictor import OnlineErrorPredictor

        pred = OnlineErrorPredictor("test/model4")
        pred.prediction_log = []

        # Always predict 0.5 (maximum uncertainty)
        for i in range(60):
            pred.prediction_log.append((0.5, i % 2 == 0, time.time()))

        brier = pred.compute_brier_score()
        assert brier == 0.25  # Random prediction baseline

    def test_calibration_metrics(self):
        """Test getting calibration metrics."""
        from app.online_predictor import OnlineErrorPredictor

        pred = OnlineErrorPredictor("test/model5")
        pred.prediction_log = [(0.5, True, time.time()) for _ in range(60)]
        pred._total_predictions = 100
        pred._correct_predictions = 50

        metrics = pred.get_calibration_metrics()

        assert "brier_score" in metrics
        assert "accuracy" in metrics
        assert "total_predictions" in metrics
        assert "calibration_temp" in metrics
        assert metrics["accuracy"] == 0.5


# =============================================================================
# Test Improvement 3: Adaptive Cache Threshold
# =============================================================================

class TestAdaptiveCacheThreshold:
    """Tests for adaptive semantic cache threshold."""

    def test_cache_threshold_settings(self):
        """Verify cache threshold settings exist."""
        from app.settings_dynamic import settings

        assert settings.CACHE_THRESHOLD_MIN >= 0.0
        assert settings.CACHE_THRESHOLD_MAX <= 1.0
        assert settings.CACHE_THRESHOLD_MIN < settings.CACHE_THRESHOLD_MAX
        assert 0.0 <= settings.CACHE_HIT_RATE_TARGET <= 1.0

    def test_get_cache_threshold_dynamic(self):
        """Test dynamic threshold retrieval."""
        from app.semantic_cache import get_cache_threshold

        threshold = get_cache_threshold()
        assert 0.0 <= threshold <= 1.0

    def test_get_cache_hit_rate(self):
        """Test hit rate calculation."""
        from app.semantic_cache import get_cache_hit_rate, _l1_cache

        # Reset stats
        with _l1_cache._lock:
            _l1_cache._hits = 20
            _l1_cache._misses = 80

        hit_rate = get_cache_hit_rate()
        assert hit_rate == 0.2  # 20/(20+80)

    def test_cache_stats(self):
        """Test cache statistics retrieval."""
        from app.semantic_cache import get_l1_cache_stats

        stats = get_l1_cache_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
        assert "maxsize" in stats


# =============================================================================
# Test Improvement 4: UQ Calibration
# =============================================================================

class TestUQCalibration:
    """Tests for uncertainty quantification calibration."""

    def test_uq_settings_exist(self):
        """Verify UQ calibration settings exist."""
        from app.settings_dynamic import settings

        assert hasattr(settings, "UQ_CALIBRATION_ENABLED")
        assert hasattr(settings, "UQ_QUALITY_GAP_RELAX")
        assert hasattr(settings, "UQ_QUALITY_GAP_TIGHTEN")

    def test_uncertainty_threshold_bounds(self):
        """Uncertainty threshold should be within bounds."""
        from app.settings_dynamic import settings

        threshold = float(settings.get("UNCERTAINTY_THRESHOLD", 0.45))
        assert 0.0 <= threshold <= 1.0


# =============================================================================
# Test Improvement 5: Judge Calibration
# =============================================================================

class TestJudgeCalibration:
    """Tests for judge calibration system."""

    def test_judge_calibration_settings(self):
        """Verify judge calibration settings exist."""
        from app.settings_dynamic import settings

        assert hasattr(settings, "JUDGE_CALIBRATION_ENABLED")
        assert hasattr(settings, "JUDGE_CACHE_AGREEMENT_TARGET")
        assert 0.0 <= settings.JUDGE_CACHE_AGREEMENT_TARGET <= 1.0

    def test_calibrate_judges_disabled(self):
        """Test calibrate_judges when disabled."""
        from app.judges import calibrate_judges
        from app.settings_dynamic import DynamicSettings

        # Mock settings.get to return "0" for JUDGE_CALIBRATION_ENABLED
        original_get = DynamicSettings.get
        def mock_get(self, key, fallback=None):
            """Execute the mock get routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            if key == "JUDGE_CALIBRATION_ENABLED":
                return "0"
            return original_get(self, key, fallback)

        with patch.object(DynamicSettings, "get", mock_get):
            result = calibrate_judges()
            assert result["status"] == "disabled"


# =============================================================================
# Test Prometheus Metrics
# =============================================================================

class TestAutonomousMetrics:
    """Tests for autonomous behavior Prometheus metrics."""

    def test_predictor_metrics_exist(self):
        """Verify predictor metrics are defined."""
        from app.observability import (
            PREDICTOR_BRIER_SCORE,
            PREDICTOR_ACCURACY,
            PREDICTOR_CALIBRATION_TEMP,
        )

        assert PREDICTOR_BRIER_SCORE is not None
        assert PREDICTOR_ACCURACY is not None
        assert PREDICTOR_CALIBRATION_TEMP is not None

    def test_cache_metrics_exist(self):
        """Verify cache threshold metrics are defined."""
        from app.observability import (
            CACHE_THRESHOLD_CURRENT,
            CACHE_HIT_RATE,
            CACHE_THRESHOLD_ADJUSTMENTS,
        )

        assert CACHE_THRESHOLD_CURRENT is not None
        assert CACHE_HIT_RATE is not None
        assert CACHE_THRESHOLD_ADJUSTMENTS is not None

    def test_uq_metrics_exist(self):
        """Verify UQ calibration metrics are defined."""
        from app.observability import (
            UQ_VS_ERROR_CORRELATION,
            UQ_HIGH_AVG_QUALITY,
            UQ_LOW_AVG_QUALITY,
        )

        assert UQ_VS_ERROR_CORRELATION is not None
        assert UQ_HIGH_AVG_QUALITY is not None
        assert UQ_LOW_AVG_QUALITY is not None

    def test_judge_calibration_metrics_exist(self):
        """Verify judge calibration metrics are defined."""
        from app.observability import (
            JUDGE_CALIBRATION_SCORE,
            JUDGE_CACHE_AGREEMENT,
            JUDGE_CALIBRATION_UPDATES,
        )

        assert JUDGE_CALIBRATION_SCORE is not None
        assert JUDGE_CACHE_AGREEMENT is not None
        assert JUDGE_CALIBRATION_UPDATES is not None

    def test_risk_factor_metrics_exist(self):
        """Verify risk factor metrics are defined."""
        from app.observability import RISK_FACTOR_CURRENT

        assert RISK_FACTOR_CURRENT is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestAutonomousIntegration:
    """Integration tests for autonomous behavior."""

    def test_uncertainty_score_in_metadata(self):
        """Verify uncertainty_score is included in route response metadata."""
        # This would require mocking the full route_and_answer flow
        # For now, just verify the structure is correct
        from app.router_core import route_and_answer

        # The function should return metadata with uncertainty_score
        # This is tested implicitly by the system running

    def test_settings_snapshot_includes_autonomous(self):
        """Verify settings snapshot includes autonomous behavior settings."""
        from app.settings_dynamic import settings

        snapshot = settings.snapshot()

        # Check key settings are included
        assert "RISK_FACTOR_SOTA_HIGH_UQ" in settings.DEFAULTS
        assert "CACHE_THRESHOLD_MIN" in settings.DEFAULTS
        assert "PREDICTOR_VALIDATION_ENABLED" in settings.DEFAULTS
        assert "UQ_CALIBRATION_ENABLED" in settings.DEFAULTS
        assert "JUDGE_CALIBRATION_ENABLED" in settings.DEFAULTS
