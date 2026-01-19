# -*- coding: utf-8 -*-
"""
Tests for query distribution drift detection.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestCosineDistance:
    """Test suite for cosine_distance function."""

    def test_identical_vectors(self):
        """Test that identical vectors have distance 0."""
        from app.app.drift_detector import cosine_distance

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])

        assert cosine_distance(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have distance 1."""
        from app.app.drift_detector import cosine_distance

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])

        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors(self):
        """Test that opposite vectors have distance 2."""
        from app.app.drift_detector import cosine_distance

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])

        assert cosine_distance(a, b) == pytest.approx(2.0, abs=1e-6)

    def test_zero_vector(self):
        """Test that zero vectors return neutral distance."""
        from app.app.drift_detector import cosine_distance

        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])

        assert cosine_distance(a, b) == 1.0


class TestQueryDriftDetector:
    """Test suite for QueryDriftDetector."""

    @pytest.fixture
    def drift_detector(self):
        """Create a fresh drift detector instance."""
        with patch("app.app.drift_detector.get_redis", return_value=None):
            with patch("app.app.drift_detector.settings") as mock_settings:
                mock_settings.DRIFT_WINDOW_SIZE = 100
                mock_settings.DRIFT_THRESHOLD = 0.15

                from app.app.drift_detector import QueryDriftDetector
                # Reset singleton
                QueryDriftDetector._instance = None
                return QueryDriftDetector()

    def test_record_query_builds_baseline(self, drift_detector):
        """Test that initial queries build baseline."""
        embedding = [0.1] * 768

        result = drift_detector.record_query(embedding)

        assert result["drift_detected"] is False
        assert "Building baseline" in result.get("message", "")

    def test_record_query_updates_total(self, drift_detector):
        """Test that recording updates total query count."""
        embedding = [0.1] * 768

        initial = drift_detector._total_queries
        drift_detector.record_query(embedding)

        assert drift_detector._total_queries == initial + 1

    def test_drift_not_detected_similar_queries(self, drift_detector):
        """Test that similar queries don't trigger drift."""
        # Build baseline with similar queries
        base_embedding = np.random.randn(768).tolist()

        # Simulate having a baseline
        drift_detector._baseline_centroid = np.array(base_embedding)
        drift_detector._baseline_sample_count = 100

        # Add similar recent embeddings
        for _ in range(20):
            similar = (np.array(base_embedding) + np.random.randn(768) * 0.01).tolist()
            drift_detector._recent_embeddings.append(np.array(similar))

        result = drift_detector.record_query(base_embedding)

        # Should not detect drift for similar queries
        assert result["drift_detected"] is False or result["drift_score"] < 0.15

    def test_get_status_structure(self, drift_detector):
        """Test that get_status returns expected structure."""
        status = drift_detector.get_status()

        assert "total_queries" in status
        assert "drift_events" in status
        assert "last_drift_score" in status
        assert "threshold" in status
        assert "window_size" in status
        assert "baseline_ready" in status

    def test_reset_baseline(self, drift_detector):
        """Test that reset_baseline clears data."""
        # Set some data
        drift_detector._baseline_centroid = np.array([1.0, 0.0, 0.0])
        drift_detector._baseline_sample_count = 100

        drift_detector.reset_baseline()

        assert drift_detector._baseline_centroid is None
        assert drift_detector._baseline_sample_count == 0
        assert len(drift_detector._recent_embeddings) == 0

    def test_force_baseline_update_not_enough_samples(self, drift_detector):
        """Test that force update fails with few samples."""
        result = drift_detector.force_baseline_update()

        assert result["success"] is False

    def test_force_baseline_update_with_samples(self, drift_detector):
        """Test that force update succeeds with enough samples."""
        # Add enough recent embeddings
        for _ in range(15):
            drift_detector._recent_embeddings.append(np.random.randn(768))

        result = drift_detector.force_baseline_update()

        assert result["success"] is True
        assert drift_detector._baseline_centroid is not None


class TestGetDriftDetector:
    """Test the get_drift_detector factory function."""

    def test_returns_singleton(self):
        """Test that get_drift_detector returns singleton instance."""
        with patch("app.app.drift_detector.get_redis", return_value=None):
            with patch("app.app.drift_detector.settings") as mock_settings:
                mock_settings.DRIFT_WINDOW_SIZE = 100
                mock_settings.DRIFT_THRESHOLD = 0.15

                from app.app.drift_detector import get_drift_detector, QueryDriftDetector
                QueryDriftDetector._instance = None

                detector1 = get_drift_detector()
                detector2 = get_drift_detector()

                assert detector1 is detector2
