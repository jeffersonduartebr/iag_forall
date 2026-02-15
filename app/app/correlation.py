# -*- coding: utf-8 -*-
"""
correlation.py — Request Correlation ID Infrastructure
-------------------------------------------------------
Provides correlation ID utilities for end-to-end request tracing.
Uses contextvars for async-safe propagation across the request lifecycle.
"""

from __future__ import annotations

import uuid
import contextvars
from typing import Optional

# Context variable to hold the correlation ID for the current request
# This is async-safe and propagates through async/await chains
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


def generate_correlation_id() -> str:
    """Generate a new unique correlation ID (UUID4)."""
    return str(uuid.uuid4())


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set the correlation ID for the current request context.

    Args:
        correlation_id: Optional existing ID to use. If None, generates a new one.

    Returns:
        The correlation ID that was set.
    """
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    _correlation_id.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """
    Get the correlation ID for the current request context.

    Returns:
        The correlation ID or None if not set.
    """
    return _correlation_id.get()


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context."""
    _correlation_id.set(None)


class CorrelationIdContext:
    """
    Context manager for correlation ID scope.

    Usage:
        with CorrelationIdContext("my-correlation-id"):
            # All code here has access to the correlation ID
            cid = get_correlation_id()
    """

    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or generate_correlation_id()
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> str:
        self._token = _correlation_id.set(self.correlation_id)
        return self.correlation_id

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None:
            _correlation_id.reset(self._token)
        return None


# HTTP Header name for correlation ID propagation
CORRELATION_ID_HEADER = "X-Correlation-ID"
