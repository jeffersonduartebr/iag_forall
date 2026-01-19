# -*- coding: utf-8 -*-
"""
adaptive_timeout.py — Adaptive Timeout Calculation
---------------------------------------------------
Calculates dynamic timeouts for LLM requests based on:
- Historical EMA latency data
- Model type (reasoning vs standard)
- Variance buffer for reliability

Features:
- Uses EMA latency history when available
- Applies configurable multipliers
- Adds variance buffer for reliability
- Clamps to min/max bounds
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from .settings_dynamic import settings

logger = logging.getLogger(__name__)

# Model markers that indicate reasoning/thinking models (typically slower)
REASONING_MODEL_MARKERS = [
    "o1", "o3", "thinking", "reasoning", "reflect",
    "deepseek-r1", "claude-3-opus", "claude-opus",
]

# Model markers for fast local models
FAST_LOCAL_MARKERS = ["phi", "gemma", "llama3:8b", "mistral:7b"]


def is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning/thinking model."""
    model_lower = model.lower()
    return any(marker in model_lower for marker in REASONING_MODEL_MARKERS)


def is_fast_local_model(model: str) -> bool:
    """Check if a model is a fast local model."""
    model_lower = model.lower()
    if "ollama" not in model_lower:
        return False
    return any(marker in model_lower for marker in FAST_LOCAL_MARKERS)


def get_ema_latency(model: str, modality: str = "text") -> Optional[float]:
    """
    Get EMA latency for a model from the database.

    Returns None if no data is available.
    """
    try:
        # Import here to avoid circular imports
        from sqlalchemy import create_engine, text

        db_url = (
            f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        engine = create_engine(db_url, pool_pre_ping=True)

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT ema_latency
                    FROM ema_history
                    WHERE model = :model AND modality = :modality
                    LIMIT 1
                """),
                {"model": model, "modality": modality}
            ).fetchone()

            if result:
                return float(result[0])
    except Exception as e:
        logger.debug(f"[AdaptiveTimeout] Failed to get EMA latency for {model}: {e}")

    return None


def calculate_adaptive_timeout(
    model: str,
    modality: str = "text",
    ema_latency: Optional[float] = None,
) -> int:
    """
    Calculate an adaptive timeout for a model request.

    Args:
        model: Model name
        modality: Query modality
        ema_latency: Pre-computed EMA latency (optional)

    Returns:
        Timeout in seconds
    """
    if not settings.ADAPTIVE_TIMEOUT_ENABLED:
        # Return default timeout when disabled
        return int(settings.get("REQUEST_TIMEOUT_SECONDS", 120))

    min_timeout = settings.MIN_TIMEOUT
    max_timeout = settings.MAX_TIMEOUT

    # Get EMA latency if not provided
    if ema_latency is None:
        ema_latency = get_ema_latency(model, modality)

    # Default latency estimates if no data
    if ema_latency is None:
        if is_fast_local_model(model):
            ema_latency = 2.0  # Fast local models
        elif is_reasoning_model(model):
            ema_latency = 30.0  # Reasoning models are slow
        elif "ollama" in model.lower():
            ema_latency = 5.0  # Standard local models
        else:
            ema_latency = 10.0  # Cloud models

    # Select multiplier based on model type
    if is_reasoning_model(model):
        multiplier = settings.ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER
    else:
        multiplier = settings.ADAPTIVE_TIMEOUT_MULTIPLIER

    # Calculate base timeout
    base_timeout = ema_latency * multiplier

    # Add 20% variance buffer for reliability
    variance_buffer = base_timeout * 0.2
    calculated_timeout = base_timeout + variance_buffer

    # Clamp to bounds
    final_timeout = max(min_timeout, min(max_timeout, int(calculated_timeout)))

    logger.debug(
        f"[AdaptiveTimeout] {model}: EMA={ema_latency:.1f}s, "
        f"mult={multiplier}, calculated={calculated_timeout:.1f}s, "
        f"final={final_timeout}s"
    )

    return final_timeout


def get_timeout_for_request(
    model: str,
    modality: str = "text",
    user_timeout: Optional[int] = None,
) -> int:
    """
    Get the timeout to use for a request.

    Priority:
    1. User-specified timeout (if provided and valid)
    2. Adaptive timeout (if enabled)
    3. Default timeout

    Args:
        model: Model name
        modality: Query modality
        user_timeout: User-specified timeout override

    Returns:
        Timeout in seconds
    """
    min_timeout = settings.MIN_TIMEOUT
    max_timeout = settings.MAX_TIMEOUT

    # User override takes priority
    if user_timeout is not None:
        # Clamp user timeout to bounds
        return max(min_timeout, min(max_timeout, user_timeout))

    # Use adaptive timeout
    return calculate_adaptive_timeout(model, modality)


def get_timeout_status(model: str, modality: str = "text") -> Dict[str, Any]:
    """Get timeout calculation status for debugging."""
    ema_latency = get_ema_latency(model, modality)

    return {
        "model": model,
        "modality": modality,
        "adaptive_enabled": settings.ADAPTIVE_TIMEOUT_ENABLED,
        "ema_latency": ema_latency,
        "is_reasoning_model": is_reasoning_model(model),
        "is_fast_local": is_fast_local_model(model),
        "multiplier": (
            settings.ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER
            if is_reasoning_model(model)
            else settings.ADAPTIVE_TIMEOUT_MULTIPLIER
        ),
        "calculated_timeout": calculate_adaptive_timeout(model, modality, ema_latency),
        "min_timeout": settings.MIN_TIMEOUT,
        "max_timeout": settings.MAX_TIMEOUT,
    }
