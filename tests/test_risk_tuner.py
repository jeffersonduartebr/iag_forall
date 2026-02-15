# -*- coding: utf-8 -*-
"""
Tests for adaptive risk factor management.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestPerformanceRecord:
    """Test suite for PerformanceRecord dataclass."""

    def test_avg_quality_empty(self):
        """Test avg_quality returns default when no samples."""
        from app.risk_tuner import PerformanceRecord

        record = PerformanceRecord()
        assert record.avg_quality == 5.0

    def test_avg_quality_with_samples(self):
        """Test avg_quality calculation with samples."""
        from app.risk_tuner import PerformanceRecord

        record = PerformanceRecord(total_samples=10, quality_sum=80.0)
        assert record.avg_quality == 8.0

    def test_success_rate_empty(self):
        """Test success_rate returns default when no samples."""
        from app.risk_tuner import PerformanceRecord

        record = PerformanceRecord()
        assert record.success_rate == 0.5

    def test_success_rate_with_samples(self):
        """Test success_rate calculation with samples."""
        from app.risk_tuner import PerformanceRecord

        record = PerformanceRecord(total_samples=10, success_count=7)
        assert record.success_rate == 0.7

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        from app.risk_tuner import PerformanceRecord

        original = PerformanceRecord(
            total_samples=100,
            quality_sum=800.0,
            success_count=75,
        )

        data = original.to_dict()
        restored = PerformanceRecord.from_dict(data)

        assert restored.total_samples == original.total_samples
        assert restored.quality_sum == original.quality_sum
        assert restored.success_count == original.success_count


class TestAdaptiveRiskTuner:
    """Test suite for AdaptiveRiskTuner."""

    @pytest.fixture
    def risk_tuner(self):
        """Create a fresh risk tuner instance."""
        with patch("app.risk_tuner.get_redis", return_value=None):
            from app.risk_tuner import AdaptiveRiskTuner
            # Reset singleton
            AdaptiveRiskTuner._instance = None
            return AdaptiveRiskTuner()

    def test_get_model_type_sota(self, risk_tuner):
        """Test SOTA model detection."""
        assert risk_tuner._get_model_type("gpt-5-mini") == "sota"
        assert risk_tuner._get_model_type("claude-opus") == "sota"
        assert risk_tuner._get_model_type("gemini-3-pro-flash") == "sota"

    def test_get_model_type_local(self, risk_tuner):
        """Test local model detection."""
        assert risk_tuner._get_model_type("ollama/phi4:latest") == "local"
        assert risk_tuner._get_model_type("ollama/llama3:8b") == "local"
        assert risk_tuner._get_model_type("mistral:7b") == "local"

    def test_record_outcome_disabled(self, risk_tuner):
        """Test that recording is skipped when disabled."""
        with patch("app.risk_tuner.settings") as mock_settings:
            mock_settings.RISK_FACTOR_ADAPT_ENABLED = False

            # Should not raise, should just skip
            risk_tuner.record_outcome("test-model", 8.0, True)

    def test_record_outcome_updates_performance(self, risk_tuner):
        """Test that recording updates performance data."""
        with patch("app.risk_tuner.settings") as mock_settings:
            mock_settings.RISK_FACTOR_ADAPT_ENABLED = True

            initial_samples = risk_tuner._performance["local_high"].total_samples

            risk_tuner.record_outcome("ollama/test", 8.0, True)

            assert risk_tuner._performance["local_high"].total_samples == initial_samples + 1

    def test_calculate_adjustment_not_enough_samples(self, risk_tuner):
        """Test that adjustment returns current factor when not enough samples."""
        from app.risk_tuner import PerformanceRecord

        record = PerformanceRecord(total_samples=5)
        result = risk_tuner._calculate_adjustment(record, 1.3)

        assert result == 1.3

    def test_calculate_adjustment_clamps_to_bounds(self, risk_tuner):
        """Test that adjustment respects bounds."""
        from app.risk_tuner import PerformanceRecord, RISK_FACTOR_MIN, RISK_FACTOR_MAX

        # Create a record that would push factor way up
        record = PerformanceRecord(
            total_samples=100,
            quality_sum=100.0,  # avg quality = 1.0, very low
            success_count=0,
        )

        with patch("app.risk_tuner.settings") as mock_settings:
            mock_settings.RISK_FACTOR_ADAPT_RATE = 1.0  # High rate

            result = risk_tuner._calculate_adjustment(record, 0.5)
            assert RISK_FACTOR_MIN <= result <= RISK_FACTOR_MAX

    def test_get_status_structure(self, risk_tuner):
        """Test that get_status returns expected structure."""
        with patch("app.risk_tuner.settings") as mock_settings:
            mock_settings.RISK_FACTOR_ADAPT_ENABLED = True
            mock_settings.RISK_FACTOR_SOTA_HIGH_UQ = 1.3
            mock_settings.RISK_FACTOR_LOCAL_HIGH_UQ = 0.6
            mock_settings.RISK_FACTOR_LOCAL_LOW_UQ = 1.1

            status = risk_tuner.get_status()

            assert "enabled" in status
            assert "current_factors" in status
            assert "performance" in status
            assert "last_tune_time" in status

    def test_reset_clears_data(self, risk_tuner):
        """Test that reset clears all performance data."""
        # Add some data
        risk_tuner._performance["sota_high"].total_samples = 100

        risk_tuner.reset()

        assert risk_tuner._performance["sota_high"].total_samples == 0
        assert risk_tuner._last_tune_time == 0


class TestGetRiskTuner:
    """Test the get_risk_tuner factory function."""

    def test_returns_singleton(self):
        """Test that get_risk_tuner returns singleton instance."""
        with patch("app.risk_tuner.get_redis", return_value=None):
            from app.risk_tuner import get_risk_tuner, AdaptiveRiskTuner
            AdaptiveRiskTuner._instance = None

            tuner1 = get_risk_tuner()
            tuner2 = get_risk_tuner()

            assert tuner1 is tuner2
