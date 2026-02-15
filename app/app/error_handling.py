# -*- coding: utf-8 -*-
"""
error_handling.py — Structured Error Handling and Logging
---------------------------------------------------------
Provides centralized error handling with:
- Error categorization for better alerting
- Structured logging with correlation IDs
- Automatic metrics recording
- Retry recommendations
"""

from __future__ import annotations

import logging
import traceback
from enum import Enum
from typing import Optional, Dict, Any, Type
from dataclasses import dataclass, field
from datetime import datetime

from .correlation import get_correlation_id
from .observability import logger as structlog_logger

logger = logging.getLogger(__name__)


# ==============================================================================
# Error Categories
# ==============================================================================

class ErrorCategory(str, Enum):
    """Categories for error classification and alerting."""

    # Provider errors
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_AUTH_ERROR = "provider_auth_error"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_INVALID_RESPONSE = "provider_invalid_response"

    # Infrastructure errors
    REDIS_ERROR = "redis_error"
    DATABASE_ERROR = "database_error"
    VECTORSTORE_ERROR = "vectorstore_error"

    # Cache errors
    CACHE_READ_ERROR = "cache_read_error"
    CACHE_WRITE_ERROR = "cache_write_error"

    # Validation errors
    VALIDATION_ERROR = "validation_error"
    INVALID_INPUT = "invalid_input"

    # Circuit breaker
    CIRCUIT_OPEN = "circuit_open"

    # Internal errors
    INTERNAL_ERROR = "internal_error"
    CONFIGURATION_ERROR = "configuration_error"

    # Unknown
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Error severity levels for alerting."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ==============================================================================
# Error Classification
# ==============================================================================

@dataclass
class ErrorInfo:
    """Structured error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    error_type: str
    correlation_id: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    retry_recommended: bool = False
    retry_after_seconds: Optional[int] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "error_type": self.error_type,
            "correlation_id": self.correlation_id,
            "model": self.model,
            "provider": self.provider,
            "retry_recommended": self.retry_recommended,
            "retry_after_seconds": self.retry_after_seconds,
            "context": self.context,
            "timestamp": self.timestamp,
        }


# Exception to category mapping
EXCEPTION_CATEGORY_MAP: Dict[str, tuple[ErrorCategory, ErrorSeverity, bool]] = {
    # Timeouts - usually retryable
    "TimeoutError": (ErrorCategory.PROVIDER_TIMEOUT, ErrorSeverity.WARNING, True),
    "asyncio.TimeoutError": (ErrorCategory.PROVIDER_TIMEOUT, ErrorSeverity.WARNING, True),
    "httpx.TimeoutException": (ErrorCategory.PROVIDER_TIMEOUT, ErrorSeverity.WARNING, True),
    "httpx.ConnectTimeout": (ErrorCategory.PROVIDER_TIMEOUT, ErrorSeverity.WARNING, True),
    "httpx.ReadTimeout": (ErrorCategory.PROVIDER_TIMEOUT, ErrorSeverity.WARNING, True),
    "APITimeoutError": (ErrorCategory.PROVIDER_TIMEOUT, ErrorSeverity.WARNING, True),

    # Rate limits - retryable after delay
    "RateLimitError": (ErrorCategory.PROVIDER_RATE_LIMIT, ErrorSeverity.WARNING, True),
    "OpenAIRateLimitError": (ErrorCategory.PROVIDER_RATE_LIMIT, ErrorSeverity.WARNING, True),
    "AnthropicRateLimitError": (ErrorCategory.PROVIDER_RATE_LIMIT, ErrorSeverity.WARNING, True),
    "ResourceExhausted": (ErrorCategory.PROVIDER_RATE_LIMIT, ErrorSeverity.WARNING, True),

    # Connection errors - may be retryable
    "ConnectionError": (ErrorCategory.PROVIDER_UNAVAILABLE, ErrorSeverity.ERROR, True),
    "httpx.ConnectError": (ErrorCategory.PROVIDER_UNAVAILABLE, ErrorSeverity.ERROR, True),
    "APIConnectionError": (ErrorCategory.PROVIDER_UNAVAILABLE, ErrorSeverity.ERROR, True),

    # Auth errors - not retryable
    "AuthenticationError": (ErrorCategory.PROVIDER_AUTH_ERROR, ErrorSeverity.CRITICAL, False),
    "PermissionDeniedError": (ErrorCategory.PROVIDER_AUTH_ERROR, ErrorSeverity.CRITICAL, False),

    # Circuit breaker
    "CircuitBreakerError": (ErrorCategory.CIRCUIT_OPEN, ErrorSeverity.WARNING, False),
    "ProviderCircuitOpenError": (ErrorCategory.CIRCUIT_OPEN, ErrorSeverity.WARNING, False),
    "ProviderCallError": (ErrorCategory.PROVIDER_UNAVAILABLE, ErrorSeverity.ERROR, True),

    # Redis errors
    "redis.ConnectionError": (ErrorCategory.REDIS_ERROR, ErrorSeverity.ERROR, True),
    "redis.TimeoutError": (ErrorCategory.REDIS_ERROR, ErrorSeverity.WARNING, True),

    # Database errors
    "SQLAlchemyError": (ErrorCategory.DATABASE_ERROR, ErrorSeverity.ERROR, True),
    "OperationalError": (ErrorCategory.DATABASE_ERROR, ErrorSeverity.ERROR, True),

    # Validation
    "ValidationError": (ErrorCategory.VALIDATION_ERROR, ErrorSeverity.WARNING, False),
    "ValueError": (ErrorCategory.INVALID_INPUT, ErrorSeverity.WARNING, False),
}


def classify_exception(exc: Exception) -> tuple[ErrorCategory, ErrorSeverity, bool]:
    """
    Classify an exception into category, severity, and retry recommendation.

    Returns:
        Tuple of (ErrorCategory, ErrorSeverity, retry_recommended)
    """
    exc_type = type(exc).__name__

    # Check direct mapping
    if exc_type in EXCEPTION_CATEGORY_MAP:
        return EXCEPTION_CATEGORY_MAP[exc_type]

    # Check module-qualified name
    full_name = f"{type(exc).__module__}.{exc_type}"
    for key, value in EXCEPTION_CATEGORY_MAP.items():
        if key in full_name or key == exc_type:
            return value

    # Check if it's a subclass of known exceptions
    for base_class in type(exc).__mro__:
        base_name = base_class.__name__
        if base_name in EXCEPTION_CATEGORY_MAP:
            return EXCEPTION_CATEGORY_MAP[base_name]

    # Default to unknown
    return ErrorCategory.UNKNOWN, ErrorSeverity.ERROR, False


# ==============================================================================
# Error Logging Functions
# ==============================================================================

def log_error(
    error: Exception,
    category: Optional[ErrorCategory] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    include_traceback: bool = True,
    **context
) -> ErrorInfo:
    """
    Log an error with structured information.

    Args:
        error: The exception to log
        category: Optional category override (auto-detected if not provided)
        model: Model name if applicable
        provider: Provider name if applicable
        include_traceback: Whether to include full traceback
        **context: Additional context to include

    Returns:
        ErrorInfo object with all error details
    """
    # Auto-classify if category not provided
    if category is None:
        category, severity, retry_recommended = classify_exception(error)
    else:
        _, severity, retry_recommended = classify_exception(error)

    # Extract retry-after if available (for rate limits)
    retry_after = None
    if hasattr(error, "retry_after"):
        retry_after = int(error.retry_after)
    elif hasattr(error, "response") and hasattr(error.response, "headers"):
        retry_after_header = error.response.headers.get("Retry-After")
        if retry_after_header:
            try:
                retry_after = int(retry_after_header)
            except ValueError:
                pass

    # Build error info
    error_info = ErrorInfo(
        category=category,
        severity=severity,
        message=str(error),
        error_type=type(error).__name__,
        correlation_id=get_correlation_id(),
        model=model,
        provider=provider,
        retry_recommended=retry_recommended,
        retry_after_seconds=retry_after,
        context=context,
        traceback=traceback.format_exc() if include_traceback else None,
    )

    # Log with appropriate level
    log_func = getattr(structlog_logger, severity.value, structlog_logger.error)
    log_func(
        "error_occurred",
        **error_info.to_dict()
    )

    return error_info


def log_provider_error(
    error: Exception,
    model: str,
    operation: str = "generate",
    **context
) -> ErrorInfo:
    """
    Log a provider-specific error.

    Args:
        error: The exception
        model: Model name (e.g., "openai/gpt-4o")
        operation: Operation that failed (e.g., "generate", "embed")
        **context: Additional context

    Returns:
        ErrorInfo object
    """
    provider = model.split("/")[0] if "/" in model else "unknown"

    return log_error(
        error=error,
        model=model,
        provider=provider,
        operation=operation,
        **context
    )


def log_cache_error(
    error: Exception,
    operation: str,  # "read" or "write"
    cache_type: str = "semantic",  # "semantic", "l1", "redis"
    **context
) -> ErrorInfo:
    """Log a cache-related error."""
    category = ErrorCategory.CACHE_READ_ERROR if operation == "read" else ErrorCategory.CACHE_WRITE_ERROR

    return log_error(
        error=error,
        category=category,
        cache_type=cache_type,
        operation=operation,
        **context
    )


def log_infrastructure_error(
    error: Exception,
    component: str,  # "redis", "database", "vectorstore"
    **context
) -> ErrorInfo:
    """Log an infrastructure-related error."""
    category_map = {
        "redis": ErrorCategory.REDIS_ERROR,
        "database": ErrorCategory.DATABASE_ERROR,
        "vectorstore": ErrorCategory.VECTORSTORE_ERROR,
    }
    category = category_map.get(component, ErrorCategory.INTERNAL_ERROR)

    return log_error(
        error=error,
        category=category,
        component=component,
        **context
    )


# ==============================================================================
# Error Response Helpers
# ==============================================================================

def create_error_response(error_info: ErrorInfo) -> Dict[str, Any]:
    """
    Create a user-friendly error response from ErrorInfo.

    Returns:
        Dict suitable for API error response
    """
    # User-friendly messages by category
    user_messages = {
        ErrorCategory.PROVIDER_TIMEOUT: "The AI model is taking longer than expected. Please try again.",
        ErrorCategory.PROVIDER_RATE_LIMIT: "Too many requests. Please wait a moment and try again.",
        ErrorCategory.PROVIDER_UNAVAILABLE: "The AI service is temporarily unavailable. Please try again later.",
        ErrorCategory.PROVIDER_AUTH_ERROR: "Authentication error with AI provider. Please contact support.",
        ErrorCategory.CIRCUIT_OPEN: "This model is temporarily disabled due to errors. Trying alternative...",
        ErrorCategory.VALIDATION_ERROR: "Invalid request. Please check your input.",
        ErrorCategory.INVALID_INPUT: "Invalid input provided.",
    }

    user_message = user_messages.get(
        error_info.category,
        "An unexpected error occurred. Please try again."
    )

    response = {
        "error": True,
        "message": user_message,
        "category": error_info.category.value,
        "correlation_id": error_info.correlation_id,
        "retry_recommended": error_info.retry_recommended,
    }

    if error_info.retry_after_seconds:
        response["retry_after_seconds"] = error_info.retry_after_seconds

    return response
