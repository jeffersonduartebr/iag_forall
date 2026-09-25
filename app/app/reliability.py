# -*- coding: utf-8 -*-
# Objective: Application runtime code for reliability.
"""Application runtime code for reliability.

This module is part of the tracked codebase and should remain aligned with the
current runtime architecture and operational documentation.
"""


from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import pybreaker

from .error_handling import ErrorCategory, log_provider_error
from .model_registry import get_model_registry
from .utils.breaker_async import call_through_breaker

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
    _initialized: bool = False

    def __new__(cls) -> "ModelCircuitBreakerManager":
        """Return the process-wide singleton instance.

        The router uses one breaker manager per process so breaker state remains
        consistent across repeated calls without having to thread an instance
        through every provider integration.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize internal breaker storage once for the singleton instance."""
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
    _initialized: bool = False

    def __new__(cls) -> "RequestDeduplicator":
        """Return the process-wide request deduplicator singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize in-flight request tracking once for the singleton."""
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
            existing = self._in_flight.get(key)
            if existing is None or existing.future.done() or time.time() - existing.created_at >= self._ttl_seconds:
                existing = None
                entry = InFlightRequest(asyncio.get_running_loop().create_future(), time.time(), model, key)
                self._in_flight[key] = entry

        if existing is not None:
            # Espera fora do lock: aguardar o LLM segurando o lock serializava todas as requisições.
            try:
                return await asyncio.shield(existing.future)
            except Exception:
                return await execute_fn()  # a original falhou: executa por conta própria

        try:
            result = await execute_fn()
            if not entry.future.done():
                entry.future.set_result(result)
            return result
        except BaseException as e:
            if not entry.future.done():
                # Cancelamento vira erro comum para os seguidores (que então executam sozinhos).
                entry.future.set_exception(e if isinstance(e, Exception) else RuntimeError("requisição original cancelada"))
                entry.future.exception()  # sem seguidores, evita o log "Future exception was never retrieved"
            raise
        finally:
            async with self._request_lock:
                if self._in_flight.get(key) is entry:
                    del self._in_flight[key]

    async def cleanup_stale(self):
        """Remove expired in-flight entries that were never cleaned up.

        This is primarily a defensive maintenance hook for long-running worker
        processes. Stale entries can accumulate if callers are cancelled or if a
        request fails before normal cleanup finishes.
        """
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
    candidates: Optional[List[str]] = None,
) -> FallbackResult:
    """
    Execute a request with automatic fallback to alternative models.

    Args:
        primary_model: Primary model to use
        execute_fn: Async function that takes model name and executes request
        max_fallbacks: Maximum number of fallback attempts
        candidates: Ordered fallback models; defaults to the registry's static chain

    Returns:
        FallbackResult with the outcome
    """
    registry = get_model_registry()
    breaker_manager = get_circuit_breaker_manager()

    # Build the list of models to try
    models_to_try = [primary_model]

    # Fallbacks: the caller's ordered candidates (the router's own pool) or the static registry chain.
    if candidates is not None:
        models_to_try.extend(m for m in candidates if m != primary_model)
    else:
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
            # call_through_breaker, não breaker.call_async: a versão do
            # pybreaker é Tornado-only e levanta NameError (ver utils.breaker_async).
            result = await call_through_breaker(breaker, execute_fn, model)

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
    get_circuit_breaker_manager()

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
# Cascade Failure Detection (extraído para services/cascade_detector.py)
# ==============================================================================

from .services.cascade_detector import (  # noqa: E402,F401  (reexport p/ chamadores existentes)
    CascadeDetector,
    get_cascade_detector,
)


def reset_reliability_runtime_state() -> None:
    """
    Reset singleton runtime state (test/dev utility).
    """
    ModelCircuitBreakerManager._instance = None
    RequestDeduplicator._instance = None
    CascadeDetector._instance = None
