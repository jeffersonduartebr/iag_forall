# -*- coding: utf-8 -*-
"""
reliability.py — Reliability Patterns for LLM Providers
--------------------------------------------------------
Provides:
- Per-model circuit breakers
- Fallback chain execution
- Request deduplication
- Retry strategies
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Dict, Optional, Callable, Any, TypeVar, Tuple, List
from dataclasses import dataclass, field
from functools import wraps
import threading

import pybreaker

from .model_registry import get_model_registry, ModelConfig
from .error_handling import log_provider_error, ErrorCategory

logger = logging.getLogger(__name__)

DEFAULT_CB_FAIL_MAX = 5
DEFAULT_CB_RESET_TIMEOUT = 60


def _runtime_cb_defaults() -> Tuple[int, int]:
    """Read circuit-breaker defaults at runtime to support hot reload."""
    try:
        from .settings_dynamic import settings as dynamic_settings
        fail_max = int(dynamic_settings.CIRCUIT_BREAKER_FAIL_MAX)
        reset_timeout = int(dynamic_settings.CIRCUIT_BREAKER_RESET_TIMEOUT)
        return max(1, fail_max), max(1, reset_timeout)
    except Exception:
        return DEFAULT_CB_FAIL_MAX, DEFAULT_CB_RESET_TIMEOUT

T = TypeVar("T")


# ==============================================================================
# Per-Model Circuit Breakers
# ==============================================================================

class ModelCircuitBreakerManager:
    """
    Manages per-model circuit breakers.

    Each model gets its own circuit breaker to prevent one failing model
    from affecting others.
    """

    _instance: Optional["ModelCircuitBreakerManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelCircuitBreakerManager":
        """Executa new."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Inicializa estado interno necessário para uso da classe."""
        if self._initialized:
            return

        self._breakers: Dict[str, pybreaker.CircuitBreaker] = {}
        self._breaker_lock = threading.Lock()
        self._initialized = True

    def get_breaker(self, model_name: str) -> pybreaker.CircuitBreaker:
        """
        Get or create a circuit breaker for a model.

        Args:
            model_name: Full model name (e.g., "openai/gpt-4o")

        Returns:
            CircuitBreaker instance for the model
        """
        if model_name not in self._breakers:
            with self._breaker_lock:
                if model_name not in self._breakers:
                    # Get config from registry for custom settings
                    registry = get_model_registry()
                    config = registry.get(model_name)

                    if config:
                        fail_max = config.circuit_breaker_threshold
                        reset_timeout = config.circuit_breaker_timeout
                    else:
                        fail_max, reset_timeout = _runtime_cb_defaults()

                    self._breakers[model_name] = pybreaker.CircuitBreaker(
                        fail_max=fail_max,
                        reset_timeout=reset_timeout,
                        name=f"breaker_{model_name}",
                    )
                    logger.debug(
                        f"[CircuitBreaker] Created breaker for {model_name}: "
                        f"fail_max={fail_max}, reset_timeout={reset_timeout}s"
                    )

        return self._breakers[model_name]

    def get_status(self, model_name: str) -> Dict[str, Any]:
        """Get the current status of a model's circuit breaker."""
        if model_name not in self._breakers:
            return {"state": "not_initialized", "model": model_name}

        breaker = self._breakers[model_name]
        return {
            "model": model_name,
            "state": breaker.current_state,
            "fail_counter": breaker.fail_counter,
            "fail_max": breaker.fail_max,
            "reset_timeout": breaker.reset_timeout,
        }

    def get_all_statuses(self) -> List[Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return [self.get_status(name) for name in self._breakers.keys()]

    def reset_breaker(self, model_name: str) -> bool:
        """Manually reset a circuit breaker."""
        if model_name in self._breakers:
            # Create a new breaker to reset state
            with self._breaker_lock:
                old_breaker = self._breakers[model_name]
                self._breakers[model_name] = pybreaker.CircuitBreaker(
                    fail_max=old_breaker.fail_max,
                    reset_timeout=old_breaker.reset_timeout,
                    name=old_breaker.name,
                )
            logger.info(f"[CircuitBreaker] Reset breaker for {model_name}")
            return True
        return False

    def is_available(self, model_name: str) -> bool:
        """Check if a model is available (circuit not open)."""
        breaker = self.get_breaker(model_name)
        return breaker.current_state != "open"


def get_circuit_breaker_manager() -> ModelCircuitBreakerManager:
    """Get the global circuit breaker manager."""
    return ModelCircuitBreakerManager()


# ==============================================================================
# Request Deduplication
# ==============================================================================

@dataclass
class InFlightRequest:
    """Tracks an in-flight request."""
    future: asyncio.Future
    created_at: float
    model: str
    query_hash: str


class RequestDeduplicator:
    """
    Deduplicates identical in-flight requests.

    If the same query is sent multiple times while a request is in progress,
    subsequent requests wait for the first one to complete.
    """

    _instance: Optional["RequestDeduplicator"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "RequestDeduplicator":
        """Executa new."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Inicializa estado interno necessário para uso da classe."""
        if self._initialized:
            return

        self._in_flight: Dict[str, InFlightRequest] = {}
        self._request_lock = asyncio.Lock()
        self._ttl_seconds = 300  # Max time to keep a request in-flight
        self._initialized = True

    def _compute_key(self, query: str, model: str, **kwargs) -> str:
        """Compute a unique key for the request."""
        # Include relevant parameters in the hash
        key_parts = [query, model]
        for k, v in sorted(kwargs.items()):
            if k in ("max_tokens", "temperature", "system_prompt"):
                key_parts.append(f"{k}={v}")

        key_string = "|".join(str(p) for p in key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    async def deduplicate(
        self,
        query: str,
        model: str,
        execute_fn: Callable[[], Any],
        **kwargs
    ) -> Any:
        """
        Execute a request with deduplication.

        If an identical request is already in progress, wait for its result
        instead of making a new request.

        Args:
            query: The query text
            model: Model name
            execute_fn: Async function to execute if not deduplicated
            **kwargs: Additional parameters for key computation

        Returns:
            Result from execute_fn (either from this call or a deduplicated one)
        """
        key = self._compute_key(query, model, **kwargs)

        async with self._request_lock:
            # Check for existing in-flight request
            if key in self._in_flight:
                request = self._in_flight[key]
                # Check if it's still valid
                if time.time() - request.created_at < self._ttl_seconds:
                    logger.debug(f"[Dedup] Found in-flight request for key {key[:8]}...")
                    try:
                        return await request.future
                    except Exception:
                        # If the original request failed, we'll try again
                        pass

            # Create new future for this request
            loop = asyncio.get_event_loop()
            future = loop.create_future()

            self._in_flight[key] = InFlightRequest(
                future=future,
                created_at=time.time(),
                model=model,
                query_hash=key,
            )

        try:
            # Execute the actual request
            result = await execute_fn()

            # Set the result for any waiting requests
            if not future.done():
                future.set_result(result)

            return result

        except Exception as e:
            # Set the exception for any waiting requests
            if not future.done():
                future.set_exception(e)
            raise

        finally:
            # Clean up
            async with self._request_lock:
                if key in self._in_flight:
                    del self._in_flight[key]

    async def cleanup_stale(self):
        """Remove stale in-flight requests."""
        now = time.time()
        async with self._request_lock:
            stale_keys = [
                k for k, v in self._in_flight.items()
                if now - v.created_at > self._ttl_seconds
            ]
            for key in stale_keys:
                del self._in_flight[key]

            if stale_keys:
                logger.info(f"[Dedup] Cleaned up {len(stale_keys)} stale requests")

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "in_flight_count": len(self._in_flight),
            "ttl_seconds": self._ttl_seconds,
        }


def get_request_deduplicator() -> RequestDeduplicator:
    """Get the global request deduplicator."""
    return RequestDeduplicator()


# ==============================================================================
# Fallback Chain Execution
# ==============================================================================

@dataclass
class FallbackResult:
    """Result of a fallback chain execution."""
    success: bool
    result: Any
    model_used: str
    models_tried: List[str]
    errors: List[Dict[str, Any]] = field(default_factory=list)


async def execute_with_fallback(
    primary_model: str,
    execute_fn: Callable[[str], Any],
    max_fallbacks: int = 3,
) -> FallbackResult:
    """
    Execute a request with automatic fallback to alternative models.

    Args:
        primary_model: Primary model to use
        execute_fn: Async function that takes model name and executes request
        max_fallbacks: Maximum number of fallback attempts

    Returns:
        FallbackResult with the outcome
    """
    registry = get_model_registry()
    breaker_manager = get_circuit_breaker_manager()

    # Build the list of models to try
    models_to_try = [primary_model]

    # Add fallback models
    fallback_chain = registry.get_fallback_chain(primary_model, max_depth=max_fallbacks)
    models_to_try.extend([m.full_name for m in fallback_chain])

    models_tried = []
    errors = []

    for model in models_to_try[:max_fallbacks + 1]:
        # Check if circuit breaker allows this model
        if not breaker_manager.is_available(model):
            logger.warning(f"[Fallback] Skipping {model}: circuit breaker open")
            errors.append({
                "model": model,
                "error": "Circuit breaker open",
                "category": ErrorCategory.CIRCUIT_OPEN.value,
            })
            continue

        models_tried.append(model)
        breaker = breaker_manager.get_breaker(model)

        try:
            # Execute with circuit breaker
            result = await breaker.call_async(execute_fn, model)

            logger.info(
                f"[Fallback] Success with {model} "
                f"(tried {len(models_tried)} model(s))"
            )

            return FallbackResult(
                success=True,
                result=result,
                model_used=model,
                models_tried=models_tried,
                errors=errors,
            )

        except pybreaker.CircuitBreakerError as e:
            logger.warning(f"[Fallback] Circuit breaker prevented call to {model}")
            errors.append({
                "model": model,
                "error": str(e),
                "category": ErrorCategory.CIRCUIT_OPEN.value,
            })

        except Exception as e:
            error_info = log_provider_error(e, model, operation="fallback_execute")
            errors.append({
                "model": model,
                "error": str(e),
                "category": error_info.category.value,
            })
            logger.warning(f"[Fallback] {model} failed: {e}")

    # All models failed
    logger.error(f"[Fallback] All {len(models_tried)} models failed")

    return FallbackResult(
        success=False,
        result=None,
        model_used=models_tried[-1] if models_tried else primary_model,
        models_tried=models_tried,
        errors=errors,
    )


# ==============================================================================
# Health Check Integration
# ==============================================================================

async def check_model_health(model_name: str) -> Dict[str, Any]:
    """
    Check the health of a specific model.

    Returns:
        Dict with health status including circuit breaker state
    """
    breaker_manager = get_circuit_breaker_manager()
    registry = get_model_registry()

    config = registry.get(model_name)
    breaker_status = breaker_manager.get_status(model_name)

    return {
        "model": model_name,
        "available": breaker_manager.is_available(model_name),
        "circuit_breaker": breaker_status,
        "config": {
            "provider": config.provider.value if config else "unknown",
            "timeout": config.default_timeout if config else 60,
            "fallbacks": config.fallback_models if config else [],
        } if config else None,
    }


async def check_all_models_health() -> Dict[str, Any]:
    """Check health of all registered models."""
    registry = get_model_registry()
    breaker_manager = get_circuit_breaker_manager()

    models = registry.list_models()
    health_checks = []

    for model in models:
        health_checks.append(await check_model_health(model.full_name))

    available_count = sum(1 for h in health_checks if h["available"])

    return {
        "total_models": len(models),
        "available_models": available_count,
        "unavailable_models": len(models) - available_count,
        "models": health_checks,
    }


# ==============================================================================
# Cascade Failure Detection
# ==============================================================================

class CascadeDetector:
    """
    Detects cascade failures across circuit breakers.

    Monitors the ratio of failed models and signals severity levels
    when too many models are failing simultaneously.
    """

    # Severity thresholds based on ratio of failed models
    THRESHOLDS = {
        "warning": 0.3,    # 30% models failed
        "critical": 0.5,   # 50% models failed
        "emergency": 0.8,  # 80% models failed
    }

    # Severity level enum
    SEVERITY_NORMAL = 0
    SEVERITY_WARNING = 1
    SEVERITY_CRITICAL = 2
    SEVERITY_EMERGENCY = 3

    _instance: Optional["CascadeDetector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CascadeDetector":
        """Executa new."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Inicializa estado interno necessário para uso da classe."""
        if self._initialized:
            return

        # Load emergency fallback models from settings
        try:
            self._emergency_fallback_models = dynamic_settings.EMERGENCY_FALLBACK_MODELS
            if not self._emergency_fallback_models:
                # Default fallback if setting is empty
                self._emergency_fallback_models = [
                    "ollama/phi4:latest",
                    "ollama/gemma3:4b",
                    "ollama/llama3:8b",
                ]
        except Exception:
            self._emergency_fallback_models = [
                "ollama/phi4:latest",
                "ollama/gemma3:4b",
                "ollama/llama3:8b",
            ]
        self._initialized = True

    def get_failed_model_ratio(self) -> Tuple[float, int, int]:
        """
        Calculate the ratio of models with open circuit breakers.

        Returns:
            Tuple of (ratio, failed_count, total_count)
        """
        manager = get_circuit_breaker_manager()
        statuses = manager.get_all_statuses()

        if not statuses:
            return 0.0, 0, 0

        total = len(statuses)
        failed = sum(1 for s in statuses if s.get("state") == "open")

        ratio = failed / total if total > 0 else 0.0
        return ratio, failed, total

    def get_severity(self) -> int:
        """
        Get current cascade failure severity level.

        Returns:
            Severity level (0=normal, 1=warning, 2=critical, 3=emergency)
        """
        ratio, _, _ = self.get_failed_model_ratio()

        if ratio >= self.THRESHOLDS["emergency"]:
            return self.SEVERITY_EMERGENCY
        elif ratio >= self.THRESHOLDS["critical"]:
            return self.SEVERITY_CRITICAL
        elif ratio >= self.THRESHOLDS["warning"]:
            return self.SEVERITY_WARNING
        else:
            return self.SEVERITY_NORMAL

    def get_severity_name(self) -> str:
        """Get severity level as string."""
        severity = self.get_severity()
        names = {
            self.SEVERITY_NORMAL: "normal",
            self.SEVERITY_WARNING: "warning",
            self.SEVERITY_CRITICAL: "critical",
            self.SEVERITY_EMERGENCY: "emergency",
        }
        return names.get(severity, "unknown")

    @property
    def is_emergency_mode(self) -> bool:
        """Check if system is in emergency mode."""
        return self.get_severity() >= self.SEVERITY_EMERGENCY

    @property
    def is_degraded(self) -> bool:
        """Check if system is in any degraded state."""
        return self.get_severity() >= self.SEVERITY_WARNING

    def get_emergency_fallback(self) -> Optional[str]:
        """
        Get a fallback model for emergency routing.

        In emergency mode, returns a known-reliable local model.
        Returns None if no fallback is available.
        """
        if not self.is_emergency_mode:
            return None

        manager = get_circuit_breaker_manager()

        # Try emergency fallback models in order
        for model in self._emergency_fallback_models:
            if manager.is_available(model):
                return model

        # All fallbacks are down - return first one anyway
        # (will likely fail but maintains consistent behavior)
        return self._emergency_fallback_models[0] if self._emergency_fallback_models else None

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive cascade detection status."""
        ratio, failed, total = self.get_failed_model_ratio()
        severity = self.get_severity()

        return {
            "severity": severity,
            "severity_name": self.get_severity_name(),
            "failed_model_ratio": round(ratio, 3),
            "failed_models": failed,
            "total_models": total,
            "is_emergency_mode": self.is_emergency_mode,
            "is_degraded": self.is_degraded,
            "emergency_fallback": self.get_emergency_fallback() if self.is_emergency_mode else None,
            "thresholds": self.THRESHOLDS,
        }

    def check_and_log_warnings(self) -> Dict[str, Any]:
        """
        Check cascade status and log warnings if needed.

        Returns status dict with any warning messages.
        """
        status = self.get_status()
        warnings = []

        if status["severity"] == self.SEVERITY_EMERGENCY:
            msg = (
                f"🚨 CASCADE EMERGENCY: {status['failed_models']}/{status['total_models']} "
                f"models failing ({status['failed_model_ratio']:.0%})"
            )
            logger.critical(msg)
            warnings.append(msg)
        elif status["severity"] == self.SEVERITY_CRITICAL:
            msg = (
                f"⚠️ CASCADE CRITICAL: {status['failed_models']}/{status['total_models']} "
                f"models failing ({status['failed_model_ratio']:.0%})"
            )
            logger.error(msg)
            warnings.append(msg)
        elif status["severity"] == self.SEVERITY_WARNING:
            msg = (
                f"⚡ Cascade warning: {status['failed_models']}/{status['total_models']} "
                f"models failing ({status['failed_model_ratio']:.0%})"
            )
            logger.warning(msg)
            warnings.append(msg)

        status["warnings"] = warnings
        return status


def get_cascade_detector() -> CascadeDetector:
    """Get the global cascade detector instance."""
    return CascadeDetector()


def reset_reliability_runtime_state() -> None:
    """
    Reset singleton runtime state (test/dev utility).
    """
    ModelCircuitBreakerManager._instance = None
    RequestDeduplicator._instance = None
    CascadeDetector._instance = None
