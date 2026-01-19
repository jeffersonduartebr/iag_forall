# -*- coding: utf-8 -*-
"""
user_feedback.py — User Feedback Processing
--------------------------------------------
Processes user feedback and integrates it with the quality scoring system.

Supports feedback types:
- thumbs_up / thumbs_down -> mapped to quality scores
- rating (1-5 stars) -> scaled to quality range
- explicit_quality -> direct quality score

Blends user feedback with original quality using configurable weight.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

from .settings_dynamic import settings
from .observability import USER_FEEDBACK_RECEIVED, USER_FEEDBACK_QUALITY_AVG
from .bandits import bandit_update, compute_reward

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Supported feedback types."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"  # 1-5 stars
    EXPLICIT_QUALITY = "explicit_quality"  # Direct 0-10 score


# Feedback type to quality score mapping
FEEDBACK_QUALITY_MAP = {
    FeedbackType.THUMBS_UP: 9.0,
    FeedbackType.THUMBS_DOWN: 2.0,
}


class UserFeedbackRequest(BaseModel):
    """Request model for user feedback submission."""
    query_id: Optional[str] = Field(None, description="Optional query ID to associate feedback with")
    query: str = Field(..., description="The original query text")
    model: str = Field(..., description="Model that generated the response")
    modality: str = Field(default="text", description="Query modality")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating value (1-5) for rating feedback type")
    quality: Optional[float] = Field(None, ge=0, le=10, description="Quality score (0-10) for explicit feedback")
    comment: Optional[str] = Field(None, description="Optional user comment")
    original_quality: Optional[float] = Field(None, description="Original quality score from judge (if available)")
    latency_s: Optional[float] = Field(None, description="Original latency in seconds")
    cost: Optional[float] = Field(None, description="Original cost")


@dataclass
class ProcessedFeedback:
    """Processed feedback result."""
    user_quality: float
    blended_quality: float
    original_quality: float
    feedback_type: FeedbackType
    model: str
    reward: float


def rating_to_quality(rating: int) -> float:
    """
    Convert 1-5 star rating to 0-10 quality score.

    Mapping:
    - 1 star -> 2.0
    - 2 stars -> 4.0
    - 3 stars -> 6.0
    - 4 stars -> 8.0
    - 5 stars -> 10.0
    """
    return float(rating) * 2.0


def get_quality_from_feedback(request: UserFeedbackRequest) -> float:
    """
    Extract quality score from feedback request.

    Returns quality in 0-10 range.
    """
    if request.feedback_type == FeedbackType.THUMBS_UP:
        return FEEDBACK_QUALITY_MAP[FeedbackType.THUMBS_UP]

    elif request.feedback_type == FeedbackType.THUMBS_DOWN:
        return FEEDBACK_QUALITY_MAP[FeedbackType.THUMBS_DOWN]

    elif request.feedback_type == FeedbackType.RATING:
        if request.rating is None:
            raise ValueError("Rating value required for rating feedback type")
        return rating_to_quality(request.rating)

    elif request.feedback_type == FeedbackType.EXPLICIT_QUALITY:
        if request.quality is None:
            raise ValueError("Quality value required for explicit_quality feedback type")
        return request.quality

    else:
        raise ValueError(f"Unknown feedback type: {request.feedback_type}")


def blend_quality(user_quality: float, original_quality: float) -> float:
    """
    Blend user feedback quality with original quality.

    Uses USER_FEEDBACK_WEIGHT setting (default 0.7 for user, 0.3 for original).
    """
    user_weight = settings.USER_FEEDBACK_WEIGHT
    original_weight = 1.0 - user_weight

    blended = user_weight * user_quality + original_weight * original_quality
    return round(blended, 2)


def process_feedback(request: UserFeedbackRequest) -> ProcessedFeedback:
    """
    Process user feedback and update bandit/metrics.

    Args:
        request: User feedback request

    Returns:
        ProcessedFeedback with quality scores and reward
    """
    # Extract quality from feedback
    user_quality = get_quality_from_feedback(request)

    # Get original quality (default to neutral if not provided)
    original_quality = request.original_quality if request.original_quality is not None else 5.0

    # Blend qualities
    blended_quality = blend_quality(user_quality, original_quality)

    # Compute reward
    latency = request.latency_s if request.latency_s is not None else 1.0
    cost = request.cost if request.cost is not None else 0.001
    reward = compute_reward(request.model, blended_quality, latency, cost)

    # Update bandit with feedback-adjusted quality
    try:
        bandit_update(
            model=request.model,
            query=request.query,
            reward=reward,
            modality=request.modality,
        )
        logger.info(
            f"[UserFeedback] Updated bandit for {request.model}: "
            f"user_q={user_quality:.1f}, blended_q={blended_quality:.1f}, reward={reward:.3f}"
        )
    except Exception as e:
        logger.warning(f"[UserFeedback] Failed to update bandit: {e}")

    # Update Prometheus metrics
    USER_FEEDBACK_RECEIVED.labels(feedback_type=request.feedback_type.value).inc()
    USER_FEEDBACK_QUALITY_AVG.set(blended_quality)

    # Persist feedback to database
    try:
        _persist_feedback(request, user_quality, blended_quality, reward)
    except Exception as e:
        logger.warning(f"[UserFeedback] Failed to persist feedback: {e}")

    return ProcessedFeedback(
        user_quality=user_quality,
        blended_quality=blended_quality,
        original_quality=original_quality,
        feedback_type=request.feedback_type,
        model=request.model,
        reward=reward,
    )


def _persist_feedback(
    request: UserFeedbackRequest,
    user_quality: float,
    blended_quality: float,
    reward: float,
) -> None:
    """Persist feedback to database."""
    try:
        from sqlalchemy import create_engine, text

        db_url = (
            f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        engine = create_engine(db_url, pool_pre_ping=True)

        # Ensure table exists
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    query_id VARCHAR(64),
                    query_text TEXT,
                    model VARCHAR(255) NOT NULL,
                    modality VARCHAR(32),
                    feedback_type VARCHAR(32) NOT NULL,
                    rating INT,
                    user_quality FLOAT NOT NULL,
                    original_quality FLOAT,
                    blended_quality FLOAT NOT NULL,
                    reward FLOAT,
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_model (model),
                    INDEX idx_feedback_type (feedback_type),
                    INDEX idx_created_at (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """))

        # Insert feedback
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO user_feedback
                    (query_id, query_text, model, modality, feedback_type, rating,
                     user_quality, original_quality, blended_quality, reward, comment)
                    VALUES
                    (:query_id, :query_text, :model, :modality, :feedback_type, :rating,
                     :user_quality, :original_quality, :blended_quality, :reward, :comment)
                """),
                {
                    "query_id": request.query_id,
                    "query_text": request.query[:2000],  # Truncate if too long
                    "model": request.model,
                    "modality": request.modality,
                    "feedback_type": request.feedback_type.value,
                    "rating": request.rating,
                    "user_quality": user_quality,
                    "original_quality": request.original_quality,
                    "blended_quality": blended_quality,
                    "reward": reward,
                    "comment": request.comment[:1000] if request.comment else None,
                }
            )
    except Exception as e:
        logger.warning(f"[UserFeedback] Database persist failed: {e}")
        raise


def get_feedback_stats(model: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
    """
    Get feedback statistics for a model or all models.

    Args:
        model: Optional model filter
        hours: Number of hours to look back

    Returns:
        Dict with feedback statistics
    """
    try:
        from sqlalchemy import create_engine, text

        db_url = (
            f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        engine = create_engine(db_url, pool_pre_ping=True)

        with engine.connect() as conn:
            if model:
                result = conn.execute(
                    text("""
                        SELECT
                            COUNT(*) as total,
                            AVG(user_quality) as avg_user_quality,
                            AVG(blended_quality) as avg_blended_quality,
                            SUM(CASE WHEN feedback_type = 'thumbs_up' THEN 1 ELSE 0 END) as thumbs_up,
                            SUM(CASE WHEN feedback_type = 'thumbs_down' THEN 1 ELSE 0 END) as thumbs_down,
                            AVG(CASE WHEN feedback_type = 'rating' THEN rating ELSE NULL END) as avg_rating
                        FROM user_feedback
                        WHERE model = :model
                        AND created_at > NOW() - INTERVAL :hours HOUR
                    """),
                    {"model": model, "hours": hours}
                ).fetchone()
            else:
                result = conn.execute(
                    text("""
                        SELECT
                            COUNT(*) as total,
                            AVG(user_quality) as avg_user_quality,
                            AVG(blended_quality) as avg_blended_quality,
                            SUM(CASE WHEN feedback_type = 'thumbs_up' THEN 1 ELSE 0 END) as thumbs_up,
                            SUM(CASE WHEN feedback_type = 'thumbs_down' THEN 1 ELSE 0 END) as thumbs_down,
                            AVG(CASE WHEN feedback_type = 'rating' THEN rating ELSE NULL END) as avg_rating
                        FROM user_feedback
                        WHERE created_at > NOW() - INTERVAL :hours HOUR
                    """),
                    {"hours": hours}
                ).fetchone()

            if result:
                return {
                    "model": model,
                    "hours": hours,
                    "total_feedback": int(result[0] or 0),
                    "avg_user_quality": round(float(result[1] or 0), 2),
                    "avg_blended_quality": round(float(result[2] or 0), 2),
                    "thumbs_up": int(result[3] or 0),
                    "thumbs_down": int(result[4] or 0),
                    "avg_rating": round(float(result[5]), 2) if result[5] else None,
                }
    except Exception as e:
        logger.warning(f"[UserFeedback] Failed to get stats: {e}")

    return {
        "model": model,
        "hours": hours,
        "total_feedback": 0,
        "error": "Failed to retrieve statistics",
    }
