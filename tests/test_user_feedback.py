# -*- coding: utf-8 -*-
"""
Tests for user feedback processing.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestFeedbackQualityMapping:
    """Test suite for feedback to quality mapping."""

    def test_rating_to_quality(self):
        """Test rating to quality conversion."""
        from app.app.user_feedback import rating_to_quality

        assert rating_to_quality(1) == 2.0
        assert rating_to_quality(2) == 4.0
        assert rating_to_quality(3) == 6.0
        assert rating_to_quality(4) == 8.0
        assert rating_to_quality(5) == 10.0

    def test_thumbs_up_quality(self):
        """Test thumbs up maps to high quality."""
        from app.app.user_feedback import (
            FEEDBACK_QUALITY_MAP,
            FeedbackType,
        )

        assert FEEDBACK_QUALITY_MAP[FeedbackType.THUMBS_UP] == 9.0

    def test_thumbs_down_quality(self):
        """Test thumbs down maps to low quality."""
        from app.app.user_feedback import (
            FEEDBACK_QUALITY_MAP,
            FeedbackType,
        )

        assert FEEDBACK_QUALITY_MAP[FeedbackType.THUMBS_DOWN] == 2.0


class TestGetQualityFromFeedback:
    """Test suite for get_quality_from_feedback function."""

    def test_thumbs_up(self):
        """Test getting quality from thumbs up feedback."""
        from app.app.user_feedback import (
            get_quality_from_feedback,
            UserFeedbackRequest,
            FeedbackType,
        )

        request = UserFeedbackRequest(
            query="test",
            model="test-model",
            feedback_type=FeedbackType.THUMBS_UP,
        )

        assert get_quality_from_feedback(request) == 9.0

    def test_thumbs_down(self):
        """Test getting quality from thumbs down feedback."""
        from app.app.user_feedback import (
            get_quality_from_feedback,
            UserFeedbackRequest,
            FeedbackType,
        )

        request = UserFeedbackRequest(
            query="test",
            model="test-model",
            feedback_type=FeedbackType.THUMBS_DOWN,
        )

        assert get_quality_from_feedback(request) == 2.0

    def test_rating(self):
        """Test getting quality from rating feedback."""
        from app.app.user_feedback import (
            get_quality_from_feedback,
            UserFeedbackRequest,
            FeedbackType,
        )

        request = UserFeedbackRequest(
            query="test",
            model="test-model",
            feedback_type=FeedbackType.RATING,
            rating=4,
        )

        assert get_quality_from_feedback(request) == 8.0

    def test_rating_missing_value(self):
        """Test that rating without value raises error."""
        from app.app.user_feedback import (
            get_quality_from_feedback,
            UserFeedbackRequest,
            FeedbackType,
        )

        request = UserFeedbackRequest(
            query="test",
            model="test-model",
            feedback_type=FeedbackType.RATING,
        )

        with pytest.raises(ValueError):
            get_quality_from_feedback(request)

    def test_explicit_quality(self):
        """Test getting quality from explicit quality feedback."""
        from app.app.user_feedback import (
            get_quality_from_feedback,
            UserFeedbackRequest,
            FeedbackType,
        )

        request = UserFeedbackRequest(
            query="test",
            model="test-model",
            feedback_type=FeedbackType.EXPLICIT_QUALITY,
            quality=7.5,
        )

        assert get_quality_from_feedback(request) == 7.5


class TestBlendQuality:
    """Test suite for blend_quality function."""

    def test_blend_quality_default_weight(self):
        """Test quality blending with default weight."""
        from app.app.user_feedback import blend_quality

        with patch("app.app.user_feedback.settings") as mock_settings:
            mock_settings.USER_FEEDBACK_WEIGHT = 0.7

            result = blend_quality(user_quality=9.0, original_quality=5.0)

            # 0.7 * 9.0 + 0.3 * 5.0 = 6.3 + 1.5 = 7.8
            assert result == pytest.approx(7.8, abs=0.01)

    def test_blend_quality_full_user_weight(self):
        """Test quality blending with full user weight."""
        from app.app.user_feedback import blend_quality

        with patch("app.app.user_feedback.settings") as mock_settings:
            mock_settings.USER_FEEDBACK_WEIGHT = 1.0

            result = blend_quality(user_quality=9.0, original_quality=5.0)

            assert result == 9.0

    def test_blend_quality_full_original_weight(self):
        """Test quality blending with full original weight."""
        from app.app.user_feedback import blend_quality

        with patch("app.app.user_feedback.settings") as mock_settings:
            mock_settings.USER_FEEDBACK_WEIGHT = 0.0

            result = blend_quality(user_quality=9.0, original_quality=5.0)

            assert result == 5.0


class TestProcessFeedback:
    """Test suite for process_feedback function."""

    def test_process_feedback_returns_result(self):
        """Test that process_feedback returns ProcessedFeedback."""
        from app.app.user_feedback import (
            process_feedback,
            UserFeedbackRequest,
            FeedbackType,
            ProcessedFeedback,
        )

        with patch("app.app.user_feedback.settings") as mock_settings:
            mock_settings.USER_FEEDBACK_WEIGHT = 0.7

            with patch("app.app.user_feedback.compute_reward", return_value=0.8):
                with patch("app.app.user_feedback.bandit_update"):
                    with patch("app.app.user_feedback._persist_feedback"):
                        with patch("app.app.user_feedback.USER_FEEDBACK_RECEIVED"):
                            with patch("app.app.user_feedback.USER_FEEDBACK_QUALITY_AVG"):
                                request = UserFeedbackRequest(
                                    query="test query",
                                    model="test-model",
                                    feedback_type=FeedbackType.THUMBS_UP,
                                )

                                result = process_feedback(request)

                                assert isinstance(result, ProcessedFeedback)
                                assert result.user_quality == 9.0
                                assert result.model == "test-model"
                                assert result.feedback_type == FeedbackType.THUMBS_UP

    def test_process_feedback_updates_bandit(self):
        """Test that process_feedback calls bandit_update."""
        from app.app.user_feedback import (
            process_feedback,
            UserFeedbackRequest,
            FeedbackType,
        )

        with patch("app.app.user_feedback.settings") as mock_settings:
            mock_settings.USER_FEEDBACK_WEIGHT = 0.7

            with patch("app.app.user_feedback.compute_reward", return_value=0.8):
                with patch("app.app.user_feedback.bandit_update") as mock_bandit:
                    with patch("app.app.user_feedback._persist_feedback"):
                        with patch("app.app.user_feedback.USER_FEEDBACK_RECEIVED"):
                            with patch("app.app.user_feedback.USER_FEEDBACK_QUALITY_AVG"):
                                request = UserFeedbackRequest(
                                    query="test query",
                                    model="test-model",
                                    feedback_type=FeedbackType.THUMBS_UP,
                                )

                                process_feedback(request)

                                mock_bandit.assert_called_once()


class TestUserFeedbackRequest:
    """Test suite for UserFeedbackRequest validation."""

    def test_valid_thumbs_up(self):
        """Test valid thumbs up request."""
        from app.app.user_feedback import UserFeedbackRequest, FeedbackType

        request = UserFeedbackRequest(
            query="test",
            model="test-model",
            feedback_type=FeedbackType.THUMBS_UP,
        )

        assert request.feedback_type == FeedbackType.THUMBS_UP

    def test_valid_rating(self):
        """Test valid rating request."""
        from app.app.user_feedback import UserFeedbackRequest, FeedbackType

        request = UserFeedbackRequest(
            query="test",
            model="test-model",
            feedback_type=FeedbackType.RATING,
            rating=4,
        )

        assert request.rating == 4

    def test_rating_out_of_range(self):
        """Test that rating must be 1-5."""
        from app.app.user_feedback import UserFeedbackRequest, FeedbackType
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserFeedbackRequest(
                query="test",
                model="test-model",
                feedback_type=FeedbackType.RATING,
                rating=6,  # Invalid: > 5
            )

    def test_quality_out_of_range(self):
        """Test that quality must be 0-10."""
        from app.app.user_feedback import UserFeedbackRequest, FeedbackType
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserFeedbackRequest(
                query="test",
                model="test-model",
                feedback_type=FeedbackType.EXPLICIT_QUALITY,
                quality=15.0,  # Invalid: > 10
            )
