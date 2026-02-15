# -*- coding: utf-8 -*-
"""
Tests for cascade failure detection.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestCascadeDetector:
    """Test suite for CascadeDetector."""

    @pytest.fixture
    def cascade_detector(self):
        """Create a fresh cascade detector instance."""
        with patch("app.reliability.get_circuit_breaker_manager"):
            from app.reliability import CascadeDetector
            # Reset singleton
            CascadeDetector._instance = None
            return CascadeDetector()

    def test_severity_normal(self, cascade_detector):
        """Test normal severity when no models are failing."""
        with patch.object(cascade_detector, "get_failed_model_ratio", return_value=(0.1, 1, 10)):
            assert cascade_detector.get_severity() == cascade_detector.SEVERITY_NORMAL
            assert cascade_detector.get_severity_name() == "normal"
            assert not cascade_detector.is_emergency_mode
            assert not cascade_detector.is_degraded

    def test_severity_warning(self, cascade_detector):
        """Test warning severity when 30% of models are failing."""
        with patch.object(cascade_detector, "get_failed_model_ratio", return_value=(0.35, 4, 10)):
            assert cascade_detector.get_severity() == cascade_detector.SEVERITY_WARNING
            assert cascade_detector.get_severity_name() == "warning"
            assert not cascade_detector.is_emergency_mode
            assert cascade_detector.is_degraded

    def test_severity_critical(self, cascade_detector):
        """Test critical severity when 50% of models are failing."""
        with patch.object(cascade_detector, "get_failed_model_ratio", return_value=(0.55, 6, 10)):
            assert cascade_detector.get_severity() == cascade_detector.SEVERITY_CRITICAL
            assert cascade_detector.get_severity_name() == "critical"
            assert not cascade_detector.is_emergency_mode
            assert cascade_detector.is_degraded

    def test_severity_emergency(self, cascade_detector):
        """Test emergency severity when 80% of models are failing."""
        with patch.object(cascade_detector, "get_failed_model_ratio", return_value=(0.85, 9, 10)):
            assert cascade_detector.get_severity() == cascade_detector.SEVERITY_EMERGENCY
            assert cascade_detector.get_severity_name() == "emergency"
            assert cascade_detector.is_emergency_mode
            assert cascade_detector.is_degraded

    def test_get_emergency_fallback_not_emergency(self, cascade_detector):
        """Test that fallback returns None when not in emergency mode."""
        with patch.object(cascade_detector, "get_severity", return_value=cascade_detector.SEVERITY_WARNING):
            cascade_detector._CascadeDetector__is_emergency_mode = False
            # Mock is_emergency_mode property
            with patch.object(type(cascade_detector), "is_emergency_mode", property(lambda self: False)):
                result = cascade_detector.get_emergency_fallback()
                assert result is None

    def test_get_status_structure(self, cascade_detector):
        """Test that get_status returns expected structure."""
        with patch.object(cascade_detector, "get_failed_model_ratio", return_value=(0.2, 2, 10)):
            status = cascade_detector.get_status()

            assert "severity" in status
            assert "severity_name" in status
            assert "failed_model_ratio" in status
            assert "failed_models" in status
            assert "total_models" in status
            assert "is_emergency_mode" in status
            assert "is_degraded" in status
            assert "thresholds" in status

    def test_check_and_log_warnings_emergency(self, cascade_detector):
        """Test that emergency warnings are logged."""
        with patch.object(cascade_detector, "get_status", return_value={
            "severity": cascade_detector.SEVERITY_EMERGENCY,
            "failed_models": 9,
            "total_models": 10,
            "failed_model_ratio": 0.9,
        }):
            with patch("app.reliability.logger") as mock_logger:
                status = cascade_detector.check_and_log_warnings()
                assert len(status.get("warnings", [])) > 0

    def test_thresholds_values(self, cascade_detector):
        """Test that thresholds have expected values."""
        assert cascade_detector.THRESHOLDS["warning"] == 0.3
        assert cascade_detector.THRESHOLDS["critical"] == 0.5
        assert cascade_detector.THRESHOLDS["emergency"] == 0.8


class TestGetCascadeDetector:
    """Test the get_cascade_detector factory function."""

    def test_returns_singleton(self):
        """Test that get_cascade_detector returns singleton instance."""
        with patch("app.reliability.get_circuit_breaker_manager"):
            from app.reliability import get_cascade_detector, CascadeDetector
            CascadeDetector._instance = None

            detector1 = get_cascade_detector()
            detector2 = get_cascade_detector()

            assert detector1 is detector2
