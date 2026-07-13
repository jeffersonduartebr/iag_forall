# Objective: API layer code for feedback routes.
"""Feedback endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from ..api.auth import AuthContext, require_api_auth
from ..api.deps import require_admin
from ..observability import logger
from ..user_feedback import UserFeedbackRequest, get_feedback_stats, process_feedback

router = APIRouter()


@router.post("/feedback", tags=["Feedback"])
def submit_feedback(
    request: UserFeedbackRequest,
    _auth: AuthContext = Depends(require_api_auth),
):
    """
    Submit user feedback for a model response.

    Supports:
    - thumbs_up / thumbs_down
    - rating (1-5 stars)
    - explicit_quality (0-10 score)
    """
    try:
        result = process_feedback(request)
        return {
            "status": "accepted",
            "user_quality": result.user_quality,
            "blended_quality": result.blended_quality,
            "model": result.model,
            "reward": round(result.reward, 4),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"[feedback] Error processing feedback: {exc}")
        raise HTTPException(status_code=500, detail="Failed to process feedback") from exc


@router.get("/feedback/stats", tags=["Feedback"])
def feedback_stats(
    model: Optional[str] = None,
    hours: int = 24,
    x_admin_token: Optional[str] = Header(None),
):
    """Get feedback statistics (admin only)."""
    require_admin(x_admin_token)
    return get_feedback_stats(model=model, hours=hours)
