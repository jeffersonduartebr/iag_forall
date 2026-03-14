# -*- coding: utf-8 -*-
# Objective: Application runtime code for providers async.
"""Implement asynchronous provider calls and provider-specific runtime behavior.

This module normalizes how the router talks to local and remote model providers.
It owns the common contract for provider responses, applies retry and circuit
breaker behavior, tracks provider metrics, and contains provider-specific logic
such as adaptive timeouts, Ollama concurrency control, and reasoning-output
separation.

The rest of the application relies on this module to expose one consistent
``call_model`` path regardless of the underlying provider implementation.
"""

from __future__ import annotations

import time
import os
import logging
import asyncio
import threading
import base64
import json
import contextlib
import traceback
import re
import warnings
import subprocess
import resource
from types import SimpleNamespace
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List

# Bibliotecas de Resiliência e HTTP Async
import httpx
import pybreaker
import requests 
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# SDKs Defensivos
try:
    from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError as OpenAIRateLimitError
except ImportError:
    AsyncOpenAI = None

try:
    from anthropic import AsyncAnthropic, RateLimitError as AnthropicRateLimitError, APIStatusError
except ImportError:
    AsyncAnthropic = None

try:
    from google import genai as google_genai
except ImportError:
    google_genai = None

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
except ImportError:
    genai = None

from pydantic import BaseModel

# Utilitários de Precisão
from app.utils.token_utils import count_tokens
from app.utils.pricing import get_model_cost
from app.model_registry import is_provider_configured

# Observabilidade
from app.observability import (
    logger as structlog_logger, 
    registry, 
    PROV_REQ, PROV_ERR, PROV_OK, PROV_LAT, PROV_COST,
    PROVIDER_TIMEOUT_ERRORS, PROVIDER_RATE_LIMIT_ERRORS,
    PROVIDER_INFLIGHT_REQUESTS,
    PROVIDER_QUEUE_WAIT,
    OLLAMA_MODEL_LOAD_SECONDS,
    OLLAMA_MODEL_LOADED,
    OLLAMA_VRAM_USED_BYTES,
    OLLAMA_VRAM_TOTAL_BYTES,
    OLLAMA_VRAM_UTILIZATION_RATIO,
    OLLAMA_DYNAMIC_CONCURRENCY_LIMIT,
    OLLAMA_DYNAMIC_CONCURRENCY_ADJUSTMENTS,
    OLLAMA_DYNAMIC_CONCURRENCY_TELEMETRY_FAILURES,
    OLLAMA_DYNAMIC_CONCURRENCY_MODE,
    GENERATION_TOKENS_PER_SECOND,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from prometheus_client import generate_latest

logger = logging.getLogger("providers_async")

# ==============================================================================
# Provider Exceptions
# ==============================================================================


class ProviderCallError(Exception):
    """Represent a normalized provider failure raised to router callers.

    Provider integrations raise many library-specific exception types. This
    wrapper converts them into a stable application-level error shape with a
    category and retryability flag.
    """

    def __init__(
        self,
        model: str,
        message: str,
        *,
        category: str = "provider_unavailable",
        retryable: bool = True,
    ):
        """Initialize the provider failure with routing-friendly metadata."""
        super().__init__(message)
        self.model = model
        self.category = category
        self.retryable = retryable


class ProviderCircuitOpenError(ProviderCallError):
    """Raised when provider execution is blocked by an open circuit breaker."""

    def __init__(self, model: str, message: str):
        """Initialize a non-retryable failure caused by circuit-breaker state."""
        super().__init__(
            model=model,
            message=message,
            category="circuit_open",
            retryable=False,
        )


def _classify_provider_exception(e: Exception) -> str:
    """Normalize provider exceptions into stable categories."""
    if isinstance(e, (asyncio.TimeoutError, httpx.TimeoutException)):
        return "provider_timeout"
    if type(e).__name__ in {
        "RateLimitError",
        "OpenAIRateLimitError",
        "AnthropicRateLimitError",
        "ResourceExhausted",
    }:
        return "provider_rate_limit"
    return "provider_unavailable"


def _build_response_meta(result: "LLMResponse") -> Dict[str, Any]:
    """Convert an internal provider response into the metadata shape used upstream."""
    return {
        "latency": result.latency,
        "load_time": result.load_time,
        "cost_per_1k": result.cost,
        "quality": heuristic_quality_estimate(result.text),
        "raw_payload": result.raw_payload,
        "image_output_b64": None,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "reasoning": result.reasoning,
    }


# ==============================================================================
# 1. ESTRATÉGIA DE RETRY (RATE LIMITS)
# ==============================================================================

RETRYABLE_ERRORS = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.HTTPStatusError,
)

if AsyncOpenAI:
    RETRYABLE_ERRORS += (APITimeoutError, APIConnectionError, OpenAIRateLimitError)

if AsyncAnthropic:
    RETRYABLE_ERRORS += (AnthropicRateLimitError, APIStatusError)

if genai:
    RETRYABLE_ERRORS += (ResourceExhausted, ServiceUnavailable)

COMMON_RETRY_STRATEGY = retry(
    reraise=True,
    stop=stop_after_attempt(5), 
    wait=wait_random_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# ==============================================================================
# 2. CIRCUIT BREAKERS GLOBAIS (Configurable via settings)
# ==============================================================================

# Load circuit breaker settings from dynamic configuration
try:
    from app.settings_dynamic import settings as dynamic_settings
    CB_FAIL_MAX = dynamic_settings.CIRCUIT_BREAKER_FAIL_MAX
    CB_RESET_TIMEOUT = dynamic_settings.CIRCUIT_BREAKER_RESET_TIMEOUT
    CB_LOCAL_FAIL_MAX = dynamic_settings.CIRCUIT_BREAKER_LOCAL_FAIL_MAX
    CB_LOCAL_RESET_TIMEOUT = dynamic_settings.CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT
except Exception:
    # Fallback defaults
    CB_FAIL_MAX = 5
    CB_RESET_TIMEOUT = 60
    CB_LOCAL_FAIL_MAX = 3
    CB_LOCAL_RESET_TIMEOUT = 30

cloud_breaker = pybreaker.CircuitBreaker(
    fail_max=CB_FAIL_MAX,
    reset_timeout=CB_RESET_TIMEOUT,
    name="cloud_breaker"
)

local_breaker = pybreaker.CircuitBreaker(
    fail_max=CB_LOCAL_FAIL_MAX,
    reset_timeout=CB_LOCAL_RESET_TIMEOUT,
    name="local_breaker"
)

# ==============================================================================
# 3. HTTP CONNECTION POOLING (Performance Optimization)
# ==============================================================================

# Global HTTP client for connection reuse
_http_client: httpx.AsyncClient | None = None
_http_client_lock = asyncio.Lock()
_ollama_tags_cache: dict[str, Any] = {"models": [], "expires_at": 0.0}
_ollama_tags_cache_lock = asyncio.Lock()
_ollama_verified_models: dict[str, float] = {}
_ollama_verified_models_lock = threading.Lock()


def _normalize_ollama_model_aliases(model_name: str) -> set[str]:
    """Return the common local aliases used for one Ollama model name."""
    normalized = str(model_name or "").strip()
    if not normalized:
        return set()
    aliases = {normalized}
    if normalized.startswith("ollama/"):
        normalized = normalized.split("/", 1)[1]
        aliases.add(normalized)
    else:
        aliases.add(f"ollama/{normalized}")
    if ":" not in normalized:
        aliases.add(f"{normalized}:latest")
        aliases.add(f"ollama/{normalized}:latest")
    return {alias for alias in aliases if alias}


def _mark_ollama_models_verified(models: list[str], ttl_seconds: int) -> None:
    """Mark one or more Ollama model names as verified-present for a short TTL."""
    expires_at = time.time() + max(1, ttl_seconds)
    with _ollama_verified_models_lock:
        for model_name in models:
            for alias in _normalize_ollama_model_aliases(model_name):
                _ollama_verified_models[alias] = expires_at


def is_ollama_model_verified(model_name: str) -> bool:
    """Return whether a model is already known present without hitting `/api/tags`."""
    now = time.time()
    with _ollama_verified_models_lock:
        stale = [name for name, expires_at in _ollama_verified_models.items() if expires_at <= now]
        for name in stale:
            _ollama_verified_models.pop(name, None)
        return any(_ollama_verified_models.get(alias, 0.0) > now for alias in _normalize_ollama_model_aliases(model_name))


async def get_http_client() -> httpx.AsyncClient:
    """
    Get or create the shared async HTTP client used by provider integrations.

    The client is initialized lazily and reused across calls to avoid repeated
    connection setup overhead. A shared pool matters especially for Ollama and
    cloud SDK fallbacks that issue frequent HTTP requests.
    """
    global _http_client

    if _http_client is None or _http_client.is_closed:
        async with _http_client_lock:
            # Double-check after acquiring lock
            if _http_client is None or _http_client.is_closed:
                _http_client = httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=200,  # Optimized for high-capacity environment
                        max_keepalive_connections=50,
                    ),
                    timeout=httpx.Timeout(60.0, connect=10.0),
                    http2=True,  # Enable HTTP/2 for better performance
                )
                logger.info("[HTTP Pool] Created global HTTP client with connection pooling (200/50)")

    return _http_client


async def close_http_client():
    """Close the shared HTTP client during process shutdown."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.close()
        _http_client = None
        reset_ollama_tags_cache()
        logger.info("[HTTP Pool] Closed global HTTP client")


async def fetch_ollama_tags(*, force_refresh: bool = False, ttl_seconds: int = 600) -> list[dict[str, Any]]:
    """Return the current Ollama model list using a small shared TTL cache.

    Frequent `/api/tags` probes create avoidable socket churn and file-descriptor
    pressure under multi-process API deployments. This helper centralizes model
    discovery behind the shared HTTP client and caches the result for a short
    period so health checks, warmup code, and model-ensure helpers do not all
    hit Ollama independently.
    """
    now = time.time()
    if not force_refresh and _ollama_tags_cache["expires_at"] > now:
        return list(_ollama_tags_cache["models"])

    async with _ollama_tags_cache_lock:
        now = time.time()
        if not force_refresh and _ollama_tags_cache["expires_at"] > now:
            return list(_ollama_tags_cache["models"])

        client = await get_http_client()
        resp = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=10.0)
        resp.raise_for_status()
        models = list(resp.json().get("models", []))
        _ollama_tags_cache["models"] = models
        _ollama_tags_cache["expires_at"] = time.time() + max(1, ttl_seconds)
        _mark_ollama_models_verified([m.get("name", "") for m in models], ttl_seconds)
        return list(models)


async def probe_ollama_liveness() -> None:
    """Perform a lightweight Ollama liveness probe using the shared HTTP pool."""
    client = await get_http_client()
    resp = await client.head(f"{OLLAMA_HOST}/", timeout=5.0)
    resp.raise_for_status()


def reset_ollama_tags_cache() -> None:
    """Invalidate the in-process Ollama tags cache."""
    _ollama_tags_cache["models"] = []
    _ollama_tags_cache["expires_at"] = 0.0
    with _ollama_verified_models_lock:
        _ollama_verified_models.clear()


def log_process_file_descriptor_limit(context: str) -> None:
    """Log the current soft and hard `nofile` limits for operational diagnosis."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        logger.info("[%s] RLIMIT_NOFILE soft=%s hard=%s", context, soft, hard)
    except Exception as exc:
        logger.warning("[%s] Failed to read RLIMIT_NOFILE: %s", context, exc)


# ==============================================================================
# 4. HELPERS E CONSTANTES
# ==============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

if genai and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Lista de palavras-chave que identificam modelos de raciocínio
REASONING_MODEL_KEYWORDS = ["phi4", "reasoning", "deepseek-r1", "cot", "math"]

# Quick Win #6 & #7: defaults + runtime-reload settings
DEFAULT_OLLAMA_CONCURRENCY_LIMIT = 5
DEFAULT_OLLAMA_DYNAMIC_CONCURRENCY_ENABLED = False
DEFAULT_OLLAMA_CONCURRENCY_MIN = 1
DEFAULT_OLLAMA_CONCURRENCY_MAX = 5
DEFAULT_OLLAMA_VRAM_TARGET_UTILIZATION = 0.72
DEFAULT_OLLAMA_VRAM_HIGH_WATERMARK = 0.82
DEFAULT_OLLAMA_VRAM_LOW_WATERMARK = 0.55
DEFAULT_OLLAMA_VRAM_POLL_INTERVAL_SECONDS = 5
DEFAULT_OLLAMA_CONCURRENCY_STEP_UP = 1
DEFAULT_OLLAMA_CONCURRENCY_STEP_DOWN = 1
DEFAULT_OLLAMA_CONCURRENCY_STABLE_WINDOWS = 3
DEFAULT_OLLAMA_GPU_INDEX = 0
DEFAULT_OLLAMA_WARM_MODELS: List[str] = []
DEFAULT_OLLAMA_INTERACTIVE_WARM_MODELS: List[str] = []
DEFAULT_OLLAMA_WARMUP_GENERATE_ENABLED = True
DEFAULT_OLLAMA_LOAD_PENALTY_ENABLED = True
DEFAULT_OLLAMA_BACKGROUND_LOAD_SHEDDING_ENABLED = True
DEFAULT_OLLAMA_ROUTE_CANDIDATE_LIMIT = 3
DEFAULT_PROVIDER_UNAVAILABLE_NEGATIVE_CACHE_TTL_SECONDS = 30
DEFAULT_BACKGROUND_THROTTLE_INFLIGHT_THRESHOLD = 4
DEFAULT_BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS = 750
DEFAULT_ADAPTIVE_TIMEOUT_ENABLED = True
DEFAULT_BASE_TIMEOUT = 60
DEFAULT_MAX_TIMEOUT = 1200
DEFAULT_TIMEOUT_MULTIPLIER = 2.0
DEFAULT_REASONING_MULTIPLIER = 3.0

# Backward-compatible exported values (used by tests/legacy code)
OLLAMA_CONCURRENCY_LIMIT = DEFAULT_OLLAMA_CONCURRENCY_LIMIT
OLLAMA_DYNAMIC_CONCURRENCY_ENABLED = DEFAULT_OLLAMA_DYNAMIC_CONCURRENCY_ENABLED
ADAPTIVE_TIMEOUT_ENABLED = DEFAULT_ADAPTIVE_TIMEOUT_ENABLED
BASE_TIMEOUT = DEFAULT_BASE_TIMEOUT
MAX_TIMEOUT = DEFAULT_MAX_TIMEOUT
TIMEOUT_MULTIPLIER = DEFAULT_TIMEOUT_MULTIPLIER
REASONING_MULTIPLIER = DEFAULT_REASONING_MULTIPLIER


def _runtime_provider_settings() -> Dict[str, Any]:
    """Read runtime-adjustable provider settings with safe defaults.

    This helper supports hot-reload behavior by consulting dynamic settings at
    call time instead of relying only on import-time constants.
    """
    out = {
        "ollama_concurrency_limit": DEFAULT_OLLAMA_CONCURRENCY_LIMIT,
        "ollama_dynamic_concurrency_enabled": DEFAULT_OLLAMA_DYNAMIC_CONCURRENCY_ENABLED,
        "ollama_concurrency_min": DEFAULT_OLLAMA_CONCURRENCY_MIN,
        "ollama_concurrency_max": DEFAULT_OLLAMA_CONCURRENCY_MAX,
        "ollama_vram_target_utilization": DEFAULT_OLLAMA_VRAM_TARGET_UTILIZATION,
        "ollama_vram_high_watermark": DEFAULT_OLLAMA_VRAM_HIGH_WATERMARK,
        "ollama_vram_low_watermark": DEFAULT_OLLAMA_VRAM_LOW_WATERMARK,
        "ollama_vram_poll_interval_seconds": DEFAULT_OLLAMA_VRAM_POLL_INTERVAL_SECONDS,
        "ollama_concurrency_step_up": DEFAULT_OLLAMA_CONCURRENCY_STEP_UP,
        "ollama_concurrency_step_down": DEFAULT_OLLAMA_CONCURRENCY_STEP_DOWN,
        "ollama_concurrency_stable_windows": DEFAULT_OLLAMA_CONCURRENCY_STABLE_WINDOWS,
        "ollama_gpu_index": DEFAULT_OLLAMA_GPU_INDEX,
        "ollama_warm_models": list(DEFAULT_OLLAMA_WARM_MODELS),
        "ollama_interactive_warm_models": list(DEFAULT_OLLAMA_INTERACTIVE_WARM_MODELS),
        "ollama_warmup_generate_enabled": DEFAULT_OLLAMA_WARMUP_GENERATE_ENABLED,
        "ollama_load_penalty_enabled": DEFAULT_OLLAMA_LOAD_PENALTY_ENABLED,
        "ollama_background_load_shedding_enabled": DEFAULT_OLLAMA_BACKGROUND_LOAD_SHEDDING_ENABLED,
        "ollama_route_candidate_limit": DEFAULT_OLLAMA_ROUTE_CANDIDATE_LIMIT,
        "provider_unavailable_negative_cache_ttl_seconds": DEFAULT_PROVIDER_UNAVAILABLE_NEGATIVE_CACHE_TTL_SECONDS,
        "background_throttle_inflight_threshold": DEFAULT_BACKGROUND_THROTTLE_INFLIGHT_THRESHOLD,
        "background_throttle_queue_wait_p95_ms": DEFAULT_BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS,
        "adaptive_timeout_enabled": DEFAULT_ADAPTIVE_TIMEOUT_ENABLED,
        "base_timeout": DEFAULT_BASE_TIMEOUT,
        "max_timeout": DEFAULT_MAX_TIMEOUT,
        "timeout_multiplier": DEFAULT_TIMEOUT_MULTIPLIER,
        "reasoning_multiplier": DEFAULT_REASONING_MULTIPLIER,
    }
    try:
        from app.settings_dynamic import settings as dynamic_settings
        out["ollama_concurrency_limit"] = max(1, int(dynamic_settings.get("OLLAMA_CONCURRENCY_LIMIT", str(DEFAULT_OLLAMA_CONCURRENCY_LIMIT))))
        out["ollama_dynamic_concurrency_enabled"] = str(dynamic_settings.get("OLLAMA_DYNAMIC_CONCURRENCY_ENABLED", "0")).strip() in ("1", "true", "True")
        out["ollama_concurrency_min"] = max(1, int(dynamic_settings.get("OLLAMA_CONCURRENCY_MIN", str(DEFAULT_OLLAMA_CONCURRENCY_MIN))))
        out["ollama_concurrency_max"] = max(out["ollama_concurrency_min"], int(dynamic_settings.get("OLLAMA_CONCURRENCY_MAX", str(out["ollama_concurrency_limit"]))))
        out["ollama_vram_target_utilization"] = min(0.99, max(0.1, float(dynamic_settings.get("OLLAMA_VRAM_TARGET_UTILIZATION", str(DEFAULT_OLLAMA_VRAM_TARGET_UTILIZATION)))))
        out["ollama_vram_high_watermark"] = min(0.99, max(0.1, float(dynamic_settings.get("OLLAMA_VRAM_HIGH_WATERMARK", str(DEFAULT_OLLAMA_VRAM_HIGH_WATERMARK)))))
        out["ollama_vram_low_watermark"] = min(out["ollama_vram_high_watermark"], max(0.0, float(dynamic_settings.get("OLLAMA_VRAM_LOW_WATERMARK", str(DEFAULT_OLLAMA_VRAM_LOW_WATERMARK)))))
        out["ollama_vram_poll_interval_seconds"] = max(1, int(dynamic_settings.get("OLLAMA_VRAM_POLL_INTERVAL_SECONDS", str(DEFAULT_OLLAMA_VRAM_POLL_INTERVAL_SECONDS))))
        out["ollama_concurrency_step_up"] = max(1, int(dynamic_settings.get("OLLAMA_CONCURRENCY_STEP_UP", str(DEFAULT_OLLAMA_CONCURRENCY_STEP_UP))))
        out["ollama_concurrency_step_down"] = max(1, int(dynamic_settings.get("OLLAMA_CONCURRENCY_STEP_DOWN", str(DEFAULT_OLLAMA_CONCURRENCY_STEP_DOWN))))
        out["ollama_concurrency_stable_windows"] = max(1, int(dynamic_settings.get("OLLAMA_CONCURRENCY_STABLE_WINDOWS", str(DEFAULT_OLLAMA_CONCURRENCY_STABLE_WINDOWS))))
        out["ollama_gpu_index"] = max(0, int(dynamic_settings.get("OLLAMA_GPU_INDEX", str(DEFAULT_OLLAMA_GPU_INDEX))))
        warm_models_raw = dynamic_settings.get("OLLAMA_WARM_MODELS", "[]")
        if isinstance(warm_models_raw, str):
            try:
                parsed_warm_models = json.loads(warm_models_raw)
            except Exception:
                parsed_warm_models = [item.strip() for item in warm_models_raw.split(",") if item.strip()]
        elif isinstance(warm_models_raw, (list, tuple, set)):
            parsed_warm_models = list(warm_models_raw)
        else:
            parsed_warm_models = []
        out["ollama_warm_models"] = [
            (str(model).strip() if str(model).strip().startswith("ollama/") else f"ollama/{str(model).strip()}")
            for model in parsed_warm_models
            if str(model).strip()
        ]
        interactive_warm_models_raw = dynamic_settings.get("OLLAMA_INTERACTIVE_WARM_MODELS", "[]")
        interactive_model_set_raw = dynamic_settings.get("OLLAMA_INTERACTIVE_MODEL_SET", "[]")
        if str(interactive_model_set_raw).strip() not in {"", "[]"}:
            interactive_warm_models_raw = interactive_model_set_raw
        if isinstance(interactive_warm_models_raw, str):
            try:
                parsed_interactive_warm_models = json.loads(interactive_warm_models_raw)
            except Exception:
                parsed_interactive_warm_models = [item.strip() for item in interactive_warm_models_raw.split(",") if item.strip()]
        elif isinstance(interactive_warm_models_raw, (list, tuple, set)):
            parsed_interactive_warm_models = list(interactive_warm_models_raw)
        else:
            parsed_interactive_warm_models = []
        out["ollama_interactive_warm_models"] = [
            (str(model).strip() if str(model).strip().startswith("ollama/") else f"ollama/{str(model).strip()}")
            for model in parsed_interactive_warm_models
            if str(model).strip()
        ]
        out["ollama_warmup_generate_enabled"] = str(dynamic_settings.get("OLLAMA_WARMUP_GENERATE_ENABLED", "1")).strip() in ("1", "true", "True")
        out["ollama_load_penalty_enabled"] = str(dynamic_settings.get("OLLAMA_LOAD_PENALTY_ENABLED", "1")).strip() in ("1", "true", "True")
        out["ollama_background_load_shedding_enabled"] = str(dynamic_settings.get("OLLAMA_BACKGROUND_LOAD_SHEDDING_ENABLED", "1")).strip() in ("1", "true", "True")
        out["ollama_route_candidate_limit"] = max(2, int(dynamic_settings.get("OLLAMA_ROUTE_CANDIDATE_LIMIT", str(DEFAULT_OLLAMA_ROUTE_CANDIDATE_LIMIT))))
        out["provider_unavailable_negative_cache_ttl_seconds"] = max(
            0,
            int(
                dynamic_settings.get(
                    "PROVIDER_UNAVAILABLE_NEGATIVE_CACHE_TTL_SECONDS",
                    dynamic_settings.get(
                        "MODEL_UNAVAILABLE_NEGATIVE_CACHE_TTL_SECONDS",
                        str(DEFAULT_PROVIDER_UNAVAILABLE_NEGATIVE_CACHE_TTL_SECONDS),
                    ),
                )
            ),
        )
        out["background_throttle_inflight_threshold"] = max(1, int(dynamic_settings.get("BACKGROUND_THROTTLE_INFLIGHT_THRESHOLD", str(DEFAULT_BACKGROUND_THROTTLE_INFLIGHT_THRESHOLD))))
        out["background_throttle_queue_wait_p95_ms"] = max(1, int(dynamic_settings.get("BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS", str(DEFAULT_BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS))))
        out["adaptive_timeout_enabled"] = str(dynamic_settings.get("ADAPTIVE_TIMEOUT_ENABLED", "1")).strip() in ("1", "true", "True")
        out["base_timeout"] = max(1, int(dynamic_settings.get("MIN_TIMEOUT", str(DEFAULT_BASE_TIMEOUT))))
        out["max_timeout"] = max(out["base_timeout"], int(dynamic_settings.get("MAX_TIMEOUT", str(DEFAULT_MAX_TIMEOUT))))
        out["timeout_multiplier"] = max(1.0, float(dynamic_settings.get("ADAPTIVE_TIMEOUT_MULTIPLIER", str(DEFAULT_TIMEOUT_MULTIPLIER))))
        out["reasoning_multiplier"] = max(1.0, float(dynamic_settings.get("ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER", str(DEFAULT_REASONING_MULTIPLIER))))
    except Exception:
        pass
    return out


class OllamaConcurrencyController:
    """Track VRAM pressure and compute a conservative dynamic Ollama concurrency limit."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._effective_limit = DEFAULT_OLLAMA_CONCURRENCY_LIMIT
        self._last_limit = DEFAULT_OLLAMA_CONCURRENCY_LIMIT
        self._latest_vram_used = 0
        self._latest_vram_total = 0
        self._latest_ratio = 0.0
        self._telemetry_ok = False
        self._stable_low_windows = 0

    async def start(self) -> None:
        """Start the background polling loop if dynamic mode is enabled."""
        cfg = _runtime_provider_settings()
        fallback_limit = int(cfg["ollama_concurrency_limit"])
        self._effective_limit = fallback_limit
        self._last_limit = fallback_limit
        if not bool(cfg["ollama_dynamic_concurrency_enabled"]):
            self._publish_mode(dynamic=False)
            self._publish_limit(fallback_limit)
            return

        async with self._lock:
            if self._task and not self._task.done():
                return
            self._stop_event = asyncio.Event()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the background polling loop."""
        async with self._lock:
            if not self._task:
                return
            self._stop_event.set()
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def get_effective_limit(self) -> int:
        """Return the latest effective concurrency limit."""
        cfg = _runtime_provider_settings()
        fallback_limit = int(cfg["ollama_concurrency_limit"])
        if not bool(cfg["ollama_dynamic_concurrency_enabled"]):
            self._publish_mode(dynamic=False)
            self._publish_limit(fallback_limit)
            return fallback_limit
        return max(1, int(self._effective_limit or fallback_limit))

    async def force_refresh(self) -> int:
        """Run one telemetry refresh cycle synchronously and return the new limit."""
        await self._refresh_once()
        return self.get_effective_limit()

    async def _run(self) -> None:
        """Continuously refresh VRAM telemetry and the derived concurrency limit."""
        while not self._stop_event.is_set():
            await self._refresh_once()
            cfg = _runtime_provider_settings()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=float(cfg["ollama_vram_poll_interval_seconds"]),
                )
            except asyncio.TimeoutError:
                continue

    async def _refresh_once(self) -> None:
        """Read VRAM telemetry once and update the effective concurrency limit."""
        cfg = _runtime_provider_settings()
        fallback_limit = int(cfg["ollama_concurrency_limit"])
        if not bool(cfg["ollama_dynamic_concurrency_enabled"]):
            self._effective_limit = fallback_limit
            self._publish_mode(dynamic=False)
            self._publish_limit(fallback_limit)
            return

        reading = await asyncio.to_thread(self._read_vram_snapshot, int(cfg["ollama_gpu_index"]))
        if reading is None:
            OLLAMA_DYNAMIC_CONCURRENCY_TELEMETRY_FAILURES.inc()
            self._telemetry_ok = False
            self._stable_low_windows = 0
            self._effective_limit = self._effective_limit or fallback_limit
            self._publish_mode(dynamic=False)
            self._publish_limit(self._effective_limit)
            return

        used_bytes, total_bytes = reading
        ratio = (used_bytes / total_bytes) if total_bytes > 0 else 0.0
        self._latest_vram_used = used_bytes
        self._latest_vram_total = total_bytes
        self._latest_ratio = ratio
        self._telemetry_ok = True
        self._publish_mode(dynamic=True)
        self._publish_vram(used_bytes, total_bytes, ratio)

        current_limit = max(int(cfg["ollama_concurrency_min"]), min(int(cfg["ollama_concurrency_max"]), int(self._effective_limit or fallback_limit)))
        high_watermark = float(cfg["ollama_vram_high_watermark"])
        low_watermark = float(cfg["ollama_vram_low_watermark"])
        step_up = int(cfg["ollama_concurrency_step_up"])
        step_down = int(cfg["ollama_concurrency_step_down"])
        stable_windows = int(cfg["ollama_concurrency_stable_windows"])
        min_limit = int(cfg["ollama_concurrency_min"])
        max_limit = min(int(cfg["ollama_concurrency_max"]), fallback_limit)

        new_limit = current_limit
        direction: str | None = None
        if ratio >= high_watermark:
            new_limit = max(min_limit, current_limit - step_down)
            self._stable_low_windows = 0
            direction = "down" if new_limit != current_limit else None
        elif ratio <= low_watermark:
            self._stable_low_windows += 1
            if self._stable_low_windows >= stable_windows:
                new_limit = min(max_limit, current_limit + step_up)
                self._stable_low_windows = 0
                direction = "up" if new_limit != current_limit else None
        else:
            self._stable_low_windows = 0

        self._effective_limit = new_limit
        self._publish_limit(new_limit)
        if direction:
            OLLAMA_DYNAMIC_CONCURRENCY_ADJUSTMENTS.labels(direction=direction).inc()
            logger.info(
                "[ollama] Dynamic concurrency adjusted from %s to %s (vram_used=%s, vram_total=%s, ratio=%.3f)",
                current_limit,
                new_limit,
                used_bytes,
                total_bytes,
                ratio,
            )

    def _read_vram_snapshot(self, gpu_index: int) -> tuple[int, int] | None:
        """Return one `(used_bytes, total_bytes)` snapshot from `nvidia-smi`."""
        cmd = [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            line = (proc.stdout or "").strip().splitlines()[0]
            used_mb_str, total_mb_str = [part.strip() for part in line.split(",", 1)]
            return int(used_mb_str) * 1024 * 1024, int(total_mb_str) * 1024 * 1024
        except Exception:
            return None

    def _publish_vram(self, used_bytes: int, total_bytes: int, ratio: float) -> None:
        """Publish current VRAM telemetry gauges."""
        try:
            OLLAMA_VRAM_USED_BYTES.set(used_bytes)
            OLLAMA_VRAM_TOTAL_BYTES.set(total_bytes)
            OLLAMA_VRAM_UTILIZATION_RATIO.set(ratio)
        except Exception:
            pass

    def _publish_limit(self, limit: int) -> None:
        """Publish current effective dynamic limit."""
        try:
            OLLAMA_DYNAMIC_CONCURRENCY_LIMIT.set(limit)
        except Exception:
            pass

    def _publish_mode(self, *, dynamic: bool) -> None:
        """Publish whether dynamic mode is active or static fallback is in effect."""
        try:
            OLLAMA_DYNAMIC_CONCURRENCY_MODE.set(1 if dynamic else 0)
        except Exception:
            pass


_ollama_concurrency_controller = OllamaConcurrencyController()
_ollama_runtime_state: Dict[str, Dict[str, Any]] = {}
_provider_unavailable_until: Dict[str, float] = {}


def _normalize_ollama_model_name(model_name: str) -> str:
    """Return one canonical Ollama model label with the `ollama/` namespace."""
    cleaned = str(model_name or "").strip()
    if not cleaned:
        return "ollama/unknown"
    return cleaned if cleaned.startswith("ollama/") else f"ollama/{cleaned}"


def _ensure_ollama_runtime_entry(model_name: str) -> Dict[str, Any]:
    """Return mutable runtime state for one Ollama model, creating defaults on first use."""
    normalized = _normalize_ollama_model_name(model_name)
    return _ollama_runtime_state.setdefault(
        normalized,
        {
            "loaded": False,
            "inflight": 0,
            "last_load_seconds": 0.0,
            "last_queue_wait_seconds": 0.0,
            "last_used_at": 0.0,
        },
    )


def _mark_ollama_model_state(
    model_name: str,
    *,
    loaded: bool | None = None,
    load_seconds: float | None = None,
    inflight_delta: int = 0,
    queue_wait_seconds: float | None = None,
) -> None:
    """Update one Ollama model runtime snapshot used for routing preferences."""
    state = _ensure_ollama_runtime_entry(model_name)
    if loaded is not None:
        state["loaded"] = bool(loaded)
    if load_seconds is not None:
        state["last_load_seconds"] = max(0.0, float(load_seconds))
    if inflight_delta:
        state["inflight"] = max(0, int(state.get("inflight", 0)) + inflight_delta)
    if queue_wait_seconds is not None:
        state["last_queue_wait_seconds"] = max(0.0, float(queue_wait_seconds))
    state["last_used_at"] = time.time()


def get_configured_ollama_warm_models() -> List[str]:
    """Return the configured warm-model list normalized to the Ollama namespace."""
    cfg = _runtime_provider_settings()
    return list(cfg.get("ollama_warm_models", []) or [])


def get_interactive_ollama_models() -> List[str]:
    """Return the preferred interactive warm-model set, falling back to the global warm list."""
    cfg = _runtime_provider_settings()
    interactive = list(cfg.get("ollama_interactive_warm_models", []) or [])
    return interactive or get_configured_ollama_warm_models()


def apply_ollama_performance_preferences(candidates: List[str], runtime_hints: Dict[str, Any] | None = None) -> List[str]:
    """Reorder and trim local candidates so routing prefers warm, less-saturated Ollama models."""
    cfg = _runtime_provider_settings()
    if not bool(cfg["ollama_load_penalty_enabled"]):
        return list(candidates)

    local_candidates = [candidate for candidate in candidates if str(candidate).startswith("ollama/")]
    if len(local_candidates) <= 1:
        return list(candidates)

    warm_set = set(get_configured_ollama_warm_models())
    interactive_warm_set = set(get_interactive_ollama_models())
    route_limit = max(2, int(cfg["ollama_route_candidate_limit"]))
    current_limit = max(1, int(_ollama_concurrency_controller.get_effective_limit()))
    workload_class = str((runtime_hints or {}).get("workload_class", "") or "").strip()
    interactive_priority = str((runtime_hints or {}).get("interactive_priority", "") or "").strip()

    def _score(candidate: str) -> tuple[float, int]:
        state = _ensure_ollama_runtime_entry(candidate)
        inflight = int(state.get("inflight", 0))
        loaded = bool(state.get("loaded", False))
        last_load_seconds = float(state.get("last_load_seconds", 0.0) or 0.0)
        last_queue_wait_seconds = float(state.get("last_queue_wait_seconds", 0.0) or 0.0)
        penalty = 0.0
        if not loaded:
            penalty += 7.0
        if candidate not in warm_set:
            penalty += 0.5
        if workload_class in {"simple_text", "knowledge_lookup"} and candidate not in interactive_warm_set:
            penalty += 2.5
        if interactive_priority == "high" and candidate in interactive_warm_set:
            penalty -= 1.5
        penalty += float(inflight) * 2.0
        if inflight >= max(1, current_limit - 1):
            penalty += 3.0
        penalty += min(4.0, last_queue_wait_seconds * 2.0)
        penalty += min(3.0, last_load_seconds / 5.0)
        return penalty, candidates.index(candidate)

    preferred_locals = sorted(local_candidates, key=_score)[: min(route_limit, len(local_candidates))]
    remaining = [candidate for candidate in candidates if not candidate.startswith("ollama/")]
    return preferred_locals + remaining


async def warm_ollama_model_runtime(model_name: str) -> bool:
    """Issue a tiny generate request so one local model is resident before user traffic arrives."""
    normalized = _normalize_ollama_model_name(model_name)
    short_name = normalized.split("/", 1)[1]
    try:
        client = await get_http_client()
        started_at = time.time()
        resp = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": short_name,
                "prompt": "Reply only with OK.",
                "stream": False,
                "think": False,
                "keep_alive": "15m",
                "options": {
                    "temperature": 0.0,
                    "num_predict": 4,
                    "num_ctx": 512,
                },
            },
            timeout=max(30.0, _get_adaptive_timeout(short_name)),
        )
        resp.raise_for_status()
        data = resp.json()
        load_seconds = float(data.get("load_duration", 0) or 0.0) / 1_000_000_000.0
        _mark_ollama_model_state(normalized, loaded=True, load_seconds=load_seconds)
        try:
            OLLAMA_MODEL_LOADED.labels(model=short_name).set(1)
            if load_seconds > 0:
                OLLAMA_MODEL_LOAD_SECONDS.labels(model=short_name).observe(load_seconds)
        except Exception:
            pass
        logger.info("[ollama] Warmed model %s in %.3fs", normalized, time.time() - started_at)
        return True
    except Exception as exc:
        logger.warning("[ollama] Failed to warm model %s: %s", normalized, exc)
        return False


async def start_provider_runtime_services() -> None:
    """Start background provider-side runtime services."""
    log_process_file_descriptor_limit("provider-runtime")
    await _ollama_concurrency_controller.start()


async def stop_provider_runtime_services() -> None:
    """Stop background provider-side runtime services."""
    await _ollama_concurrency_controller.stop()


def mark_provider_unavailable(model_name: str, *, ttl_seconds: int | None = None) -> None:
    """Remember a temporarily unavailable model to avoid hammering the same failing path."""
    cfg = _runtime_provider_settings()
    ttl = int(ttl_seconds if ttl_seconds is not None else cfg.get("provider_unavailable_negative_cache_ttl_seconds", cfg.get("model_unavailable_negative_cache_ttl_seconds", DEFAULT_PROVIDER_UNAVAILABLE_NEGATIVE_CACHE_TTL_SECONDS)))
    if ttl <= 0:
        return
    _provider_unavailable_until[str(model_name)] = time.time() + ttl


def clear_provider_unavailable(model_name: str) -> None:
    """Clear temporary negative-cache state after a successful call."""
    _provider_unavailable_until.pop(str(model_name), None)


def is_provider_temporarily_unavailable(model_name: str) -> bool:
    """Return whether one model is still inside the negative-cache cooldown window."""
    expires_at = float(_provider_unavailable_until.get(str(model_name), 0.0) or 0.0)
    if expires_at <= 0:
        return False
    if expires_at <= time.time():
        _provider_unavailable_until.pop(str(model_name), None)
        return False
    return True


def should_throttle_background_judge() -> bool:
    """Return whether background judge work should yield to interactive Ollama traffic."""
    cfg = _runtime_provider_settings()
    if not bool(cfg["ollama_background_load_shedding_enabled"]):
        return False
    current_limit = max(1, int(_ollama_concurrency_controller.get_effective_limit()))
    total_inflight = sum(int(state.get("inflight", 0) or 0) for state in _ollama_runtime_state.values())
    inflight_threshold = max(1, min(current_limit, int(cfg.get("background_throttle_inflight_threshold", current_limit))))
    if total_inflight >= inflight_threshold:
        return True
    if any(
        float(state.get("last_queue_wait_seconds", 0.0) or 0.0) * 1000.0 >= float(cfg.get("background_throttle_queue_wait_p95_ms", DEFAULT_BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS))
        for state in _ollama_runtime_state.values()
    ):
        return True
    ratio = float(getattr(_ollama_concurrency_controller, "_latest_ratio", 0.0) or 0.0)
    if ratio >= float(cfg["ollama_vram_high_watermark"]):
        return True
    return False


def get_ollama_admission_snapshot() -> Dict[str, float | int | str]:
    """Return a compact snapshot of Ollama pressure for adaptive admission control.

    The middleware layer cannot read Prometheus samples directly, so it uses this
    runtime snapshot derived from the same in-process state that powers provider
    throttling and warm-model selection.
    """
    cfg = _runtime_provider_settings()
    current_limit = max(1, int(_ollama_concurrency_controller.get_effective_limit()))
    total_inflight = sum(int(state.get("inflight", 0) or 0) for state in _ollama_runtime_state.values())
    max_queue_wait_ms = 0.0
    for state in _ollama_runtime_state.values():
        max_queue_wait_ms = max(max_queue_wait_ms, float(state.get("last_queue_wait_seconds", 0.0) or 0.0) * 1000.0)
    utilization = min(2.0, float(total_inflight) / float(current_limit)) if current_limit > 0 else 0.0
    vram_ratio = float(getattr(_ollama_concurrency_controller, "_latest_ratio", 0.0) or 0.0)
    elevated_utilization = float(cfg.get("ollama_vram_target_utilization", DEFAULT_OLLAMA_VRAM_TARGET_UTILIZATION))
    congested_utilization = float(cfg.get("ollama_vram_high_watermark", DEFAULT_OLLAMA_VRAM_HIGH_WATERMARK))
    pressure_state = "normal"
    if utilization >= 1.0 or max_queue_wait_ms >= float(cfg.get("background_throttle_queue_wait_p95_ms", DEFAULT_BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS)) or vram_ratio >= congested_utilization:
        pressure_state = "congested"
    elif utilization >= elevated_utilization or max_queue_wait_ms >= max(100.0, float(cfg.get("background_throttle_queue_wait_p95_ms", DEFAULT_BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS)) * 0.5):
        pressure_state = "elevated"
    return {
        "current_limit": current_limit,
        "total_inflight": total_inflight,
        "max_queue_wait_ms": round(max_queue_wait_ms, 3),
        "utilization": round(utilization, 4),
        "vram_ratio": round(vram_ratio, 4),
        "pressure_state": pressure_state,
    }


def _workload_provider_timeout_budget(workload_class: str | None) -> float | None:
    """Return the configured timeout budget for a known workload class."""
    workload = str(workload_class or "").strip().lower()
    key_map = {
        "simple_text": ("PROVIDER_TIMEOUT_SIMPLE_SECONDS", 20),
        "knowledge_lookup": ("PROVIDER_TIMEOUT_KNOWLEDGE_SECONDS", 35),
        "reasoning": ("PROVIDER_TIMEOUT_REASONING_SECONDS", 60),
        "vision": ("PROVIDER_TIMEOUT_VISION_SECONDS", 90),
    }
    key_default = key_map.get(workload)
    if not key_default:
        return None
    key, default = key_default
    try:
        from app.settings_dynamic import settings as dynamic_settings

        return max(5.0, float(dynamic_settings.get(key, default)))
    except Exception:
        return float(default)


def _get_adaptive_timeout(model: str, workload_class: str | None = None) -> float:
    """
    Calculate a timeout budget that reflects the expected cost of the model call.

    Reasoning and larger models receive a larger timeout budget so normal
    behavior does not look like failure, while simpler models keep lower limits
    to surface stalls earlier.
    """
    explicit_workload_timeout = _workload_provider_timeout_budget(workload_class)
    if explicit_workload_timeout is not None:
        return explicit_workload_timeout

    runtime_cfg = _runtime_provider_settings()
    adaptive_timeout_enabled = bool(runtime_cfg["adaptive_timeout_enabled"])
    base_timeout = int(runtime_cfg["base_timeout"])
    max_timeout = int(runtime_cfg["max_timeout"])
    timeout_multiplier = float(runtime_cfg["timeout_multiplier"])
    reasoning_multiplier = float(runtime_cfg["reasoning_multiplier"])

    if not adaptive_timeout_enabled:
        return max_timeout  # Fall back to original behavior

    model_lower = model.lower()

    # Reasoning models need more time for chain-of-thought
    if any(k in model_lower for k in REASONING_MODEL_KEYWORDS):
        timeout = base_timeout * reasoning_multiplier
    # Large models (>7B parameters indicated by name)
    elif any(size in model_lower for size in ["70b", "65b", "33b", "13b", "11b"]):
        timeout = base_timeout * timeout_multiplier
    # Standard models
    else:
        timeout = base_timeout * 1.5

    return min(timeout, max_timeout)

def heuristic_quality_estimate(text: str) -> float:
    """Estimate answer quality with a cheap 0-10 proxy for observability.

    This heuristic is intentionally simple. It provides a rough quality signal
    when provider metrics need a score before the formal judge pipeline has run.
    """
    if not text or not text.strip():
        return 0.0
    length = len(text)
    length_factor = min(1.0, length / 400.0)
    punctuation_bonus = 0.3 if any(p in text for p in [".", "!", "?"]) else 0.0
    score = 4.0 + 5.5 * length_factor + punctuation_bonus
    return max(0.0, min(10.0, round(score, 2)))

def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length when provider usage is unavailable."""
    if not text: return 0
    return max(1, len(text) // 4)

def render_metrics_response():
    """Serialize provider metrics using the shared Prometheus registry."""
    return generate_latest(registry), CONTENT_TYPE_LATEST

# ==============================================================================
# 4. MODELOS DE DADOS (ATUALIZADO)
# ==============================================================================

class LLMResponse(BaseModel):
    """Represent the normalized provider response consumed by router code.

    Every provider adapter returns this model so downstream code can work with a
    stable shape for text, timing, cost, token counts, raw payloads, and
    optional reasoning traces.
    """
    text: str
    latency: float
    load_time: float = 0.0
    cost: float
    prompt_tokens: int
    completion_tokens: int
    model_used: str
    raw_payload: Optional[str] = None
    reasoning: Optional[str] = None  # <--- NOVO CAMPO: Armazena o pensamento (CoT)

# ==============================================================================
# 5. ARQUITETURA BASE
# ==============================================================================

class BaseProvider(ABC):
    """Define the shared asynchronous interface implemented by all providers."""
    def __init__(self, name: str, concurrency_limit: int):
        """Initialize provider identity and the concurrency semaphore."""
        self.name = name
        self._concurrency_limit = max(1, int(concurrency_limit))
        self.semaphore = asyncio.Semaphore(self._concurrency_limit)

    @abstractmethod
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate a normalized model response for the given prompt."""
        pass

    async def _acquire_slot(self, model: str) -> None:
        """Wait for provider capacity and publish queue/in-flight metrics."""
        wait_started_at = time.time()
        await self.semaphore.acquire()
        waited = time.time() - wait_started_at
        if self.name == "ollama":
            _mark_ollama_model_state(model, inflight_delta=1, queue_wait_seconds=waited)
        try:
            PROVIDER_QUEUE_WAIT.labels(model=model).observe(waited)
            PROVIDER_INFLIGHT_REQUESTS.labels(model=model).inc()
        except Exception:
            pass

    def _release_slot(self, model: str) -> None:
        """Release provider capacity and publish in-flight metrics."""
        try:
            self.semaphore.release()
        finally:
            if self.name == "ollama":
                _mark_ollama_model_state(model, inflight_delta=-1)
            try:
                PROVIDER_INFLIGHT_REQUESTS.labels(model=model).dec()
            except Exception:
                pass

    def _record_metrics(self, model: str, latency: float, cost: float, success: bool):
        """Publish provider-level success, latency, and cost metrics."""
        PROV_REQ.labels(model=model).inc()
        if success:
            PROV_OK.labels(model=model).inc()
            PROV_LAT.labels(model=model).observe(latency)
            PROV_COST.labels(model=model).observe(cost)
        else:
            PROV_ERR.labels(model=model).inc()

    def _record_generation_metrics(self, model: str, completion_tokens: int, latency: float) -> None:
        """Publish generation throughput metrics for successful responses."""
        if latency <= 0 or completion_tokens <= 0:
            return
        try:
            GENERATION_TOKENS_PER_SECOND.labels(model=model).observe(completion_tokens / latency)
        except Exception:
            pass

# ==============================================================================
# 6. PROVEDOR: OPENAI
# ==============================================================================

class OpenAIProvider(BaseProvider):
    """Call OpenAI chat models through the async SDK and normalize the result."""
    def __init__(self):
        """Create the OpenAI client and configure cloud-provider concurrency."""
        if not AsyncOpenAI: raise ImportError("OpenAI SDK not installed")
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        super().__init__("openai", concurrency_limit=100)

    @COMMON_RETRY_STRATEGY
    @cloud_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        """Execute one OpenAI chat-completion request and normalize its output."""
        model = kwargs.get("model", "gpt-4o")
        temperature = kwargs.get("temperature", 0.5)
        max_tokens = kwargs.get("max_tokens", 512)
        start = time.time()

        await self._acquire_slot(model)
        try:
            content = [{"type": "text", "text": prompt}]
            if image_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                })

            api_args = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
            }

            if model.startswith("o1-") or "gpt-5" in model:
                api_args["max_completion_tokens"] = max_tokens
            else:
                api_args["max_tokens"] = max_tokens
                api_args["temperature"] = temperature

            resp = await self.client.chat.completions.create(**api_args)

            text_out = resp.choices[0].message.content or ""
            usage = resp.usage
            p_tok = usage.prompt_tokens if usage else 0
            c_tok = usage.completion_tokens if usage else 0

            cost = get_model_cost(model, p_tok, c_tok)
            latency = time.time() - start
            self._record_metrics(model, latency, cost, True)
            self._record_generation_metrics(model, c_tok, latency)

            try:
                raw_payload = json.dumps(resp.model_dump(), default=str)
            except Exception:
                raw_payload = str(resp)

            return LLMResponse(
                text=text_out, latency=latency, load_time=0.0,
                cost=cost, prompt_tokens=p_tok, completion_tokens=c_tok,
                model_used=model, raw_payload=raw_payload,
                reasoning=None,
            )
        except Exception as e:
            self._record_metrics(model, time.time()-start, 0, False)
            structlog_logger.error("openai_provider_fail", error=str(e), model=model)
            raise
        finally:
            self._release_slot(model)

# ==============================================================================
# 7. PROVEDOR: ANTHROPIC
# ==============================================================================

class AnthropicProvider(BaseProvider):
    """Call Anthropic models and normalize the response for router consumers."""
    def __init__(self):
        """Create the Anthropic client and configure cloud-provider concurrency."""
        if not AsyncAnthropic: raise ImportError("Anthropic SDK not installed")
        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        super().__init__("anthropic", concurrency_limit=50)

    @COMMON_RETRY_STRATEGY
    @cloud_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        """Execute one Anthropic request and normalize the provider payload."""
        model = kwargs.get("model", "claude-3-5-sonnet-latest")
        start = time.time()

        await self._acquire_slot(model)
        try:
            content = [{"type": "text", "text": prompt}]
            if image_b64:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}
                })

            resp = await self.client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.5),
                messages=[{"role": "user", "content": content}]
            )

            text_out = resp.content[0].text
            usage = resp.usage
            p_tok = usage.input_tokens
            c_tok = usage.output_tokens

            cost = get_model_cost(model, p_tok, c_tok)
            latency = time.time() - start
            self._record_metrics(model, latency, cost, True)
            self._record_generation_metrics(model, c_tok, latency)

            return LLMResponse(
                text=text_out, latency=latency, load_time=0.0,
                cost=cost, prompt_tokens=p_tok, completion_tokens=c_tok,
                model_used=model, raw_payload=str(resp),
                reasoning=None
            )
        except Exception as e:
            self._record_metrics(model, time.time()-start, 0, False)
            raise
        finally:
            self._release_slot(model)

# ==============================================================================
# 8. PROVEDOR: GEMINI (GOOGLE)
# ==============================================================================

class GeminiProvider(BaseProvider):
    """Call Gemini models through the available Google SDK path."""
    class GeminiAdapter:
        """Isolate SDK-specific Gemini calls behind a small compatibility layer."""

        def generate(
            self,
            model_name: str,
            prompt: str,
            image_b64: Optional[str],
            temperature: float,
            max_tokens: int,
        ):
            """Execute one Gemini generation call using the available SDK."""
            if google_genai is not None:
                client = google_genai.Client(api_key=GEMINI_API_KEY or None)
                parts = [{"text": prompt}]
                if image_b64:
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_b64,
                            }
                        }
                    )
                resp = client.models.generate_content(
                    model=model_name,
                    contents=parts,
                    config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                )
                text_out = getattr(resp, "text", "") or ""
                return SimpleNamespace(text=text_out)

            if genai is None:
                raise ImportError("No Gemini SDK available")

            gmodel = genai.GenerativeModel(model_name)
            legacy_parts = [prompt]
            if image_b64:
                legacy_parts.append({"mime_type": "image/jpeg", "data": base64.b64decode(image_b64)})
            return gmodel.generate_content(
                legacy_parts,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )

    def __init__(self):
        """Initialize the Gemini provider and its adapter."""
        if not genai: raise ImportError("Google GenAI SDK not installed")
        super().__init__("gemini", concurrency_limit=60)
        self._adapter = self.GeminiAdapter()

    @COMMON_RETRY_STRATEGY
    @cloud_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        """Execute one Gemini request and normalize the response payload."""
        model_name = kwargs.get("model", "gemini-1.5-flash")
        start = time.time()

        await self._acquire_slot(model_name)
        try:
            def _call():
                """Invoke the adapter in a worker thread-friendly callable."""
                return self._adapter.generate(
                    model_name=model_name,
                    prompt=prompt,
                    image_b64=image_b64,
                    temperature=kwargs.get("temperature", 0.5),
                    max_tokens=kwargs.get("max_tokens", 512),
                )

            resp = await asyncio.to_thread(_call)
            text_out = resp.text or ""

            p_tok = count_tokens(prompt, model_name)
            c_tok = count_tokens(text_out, model_name)
            cost = get_model_cost(model_name, p_tok, c_tok)

            latency = time.time() - start
            self._record_metrics(model_name, latency, cost, True)
            self._record_generation_metrics(model_name, c_tok, latency)

            return LLMResponse(
                text=text_out, latency=latency, load_time=0.0,
                cost=cost, prompt_tokens=p_tok, completion_tokens=c_tok,
                model_used=model_name, raw_payload=str(resp),
                reasoning=None
            )
        except Exception as e:
            self._record_metrics(model_name, time.time()-start, 0, False)
            raise
        finally:
            self._release_slot(model_name)

# ==============================================================================
# 9. PROVEDOR: OLLAMA (LOCAL & HTTPX ASYNC)
# ==============================================================================

class OllamaProvider(BaseProvider):
    """Call local Ollama models with adaptive timeout and concurrency controls.

    This provider owns the semaphore that limits concurrent local execution, the
    model-availability check, and the logic that separates reasoning traces from
    final answer text.
    """
    def __init__(self):
        """Initialize the local provider host and current concurrency settings."""
        self.host = OLLAMA_HOST
        cfg = _runtime_provider_settings()
        super().__init__("ollama", concurrency_limit=int(cfg["ollama_concurrency_limit"]))

    def _refresh_concurrency_limit(self) -> None:
        """Refresh the Ollama semaphore when runtime settings change."""
        new_limit = _ollama_concurrency_controller.get_effective_limit()
        if new_limit != self._concurrency_limit:
            self._concurrency_limit = new_limit
            self.semaphore = asyncio.Semaphore(new_limit)
            logger.info("[ollama] Updated concurrency limit to %s", new_limit)

    @COMMON_RETRY_STRATEGY
    @local_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        """Execute one local Ollama request and normalize text and reasoning output."""
        model = kwargs.get("model", "phi4:latest")
        start = time.time()
        self._refresh_concurrency_limit()

        # --- LÓGICA DE INJEÇÃO DE THINKING ---
        # Se for um modelo de raciocínio, forçamos a tag <think> no prompt
        # para garantir que o modelo saiba o que fazer, caso o template padrão não tenha.
        is_reasoning_model = any(k in model.lower() for k in REASONING_MODEL_KEYWORDS)
        final_prompt = prompt
        
        if is_reasoning_model:
            # Adiciona instrução de sistema implícita se não houver
            if "<think>" not in prompt:
                final_prompt = (
                    "You are a reasoning model. "
                    "Please output your thought process within <think> tags before your final answer.\n\n"
                    f"{prompt}"
                )

        await self._acquire_slot(model)
        try:
            payload = {
                "model": model,
                "prompt": final_prompt,
                "stream": False,
                # Avoid empty final answers on models that support a separate
                # thinking channel (for example qwen3.5) unless we explicitly
                # want reasoning output.
                "think": is_reasoning_model,
                "options": {
                    "temperature": kwargs.get("temperature", 0.5),
                    "num_predict": kwargs.get("max_tokens", 512),
                    "num_ctx": 4096
                }
            }
            if image_b64:
                payload["images"] = [image_b64]

            # Quick Win #6: Adaptive timeout based on model type
            explicit_timeout = kwargs.get("timeout_seconds")
            timeout = (
                max(1.0, float(explicit_timeout))
                if explicit_timeout is not None
                else _get_adaptive_timeout(model, workload_class=kwargs.get("workload_class"))
            )

            client = await get_http_client()
            resp = await client.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()

            raw_text = data.get("response", "").strip()
            raw_thinking = data.get("thinking", "").strip()

            reasoning = None
            text_out = raw_text

            think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
            if think_match:
                reasoning = think_match.group(1).strip()
                text_out = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            elif raw_thinking:
                reasoning = raw_thinking

            load_ns = data.get("load_duration", 0)
            load_sec = float(load_ns) / 1_000_000_000.0

            p_tok = data.get("prompt_eval_count", 0)
            c_tok = data.get("eval_count", 0)
            cost = get_model_cost(model, p_tok, c_tok)
            latency = time.time() - start

            self._record_metrics(model, latency, cost, True)
            self._record_generation_metrics(model, c_tok, latency)
            if load_sec > 0:
                try:
                    OLLAMA_MODEL_LOAD_SECONDS.labels(model=model).observe(load_sec)
                except Exception:
                    pass
            try:
                OLLAMA_MODEL_LOADED.labels(model=model).set(1)
            except Exception:
                pass
            _mark_ollama_model_state(model, loaded=True, load_seconds=load_sec)

            return LLMResponse(
                text=text_out,
                latency=latency,
                load_time=load_sec,
                cost=cost,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                model_used=model,
                raw_payload=json.dumps(data),
                reasoning=reasoning,
            )
        except Exception as e:
            self._record_metrics(model, time.time()-start, 0, False)

            error_msg = str(e)
            if isinstance(e, httpx.HTTPStatusError):
                try:
                    error_msg += f" | Body: {e.response.text}"
                except Exception as body_err:
                    error_msg += f" | Body read failed: {body_err}"

            structlog_logger.error(
                "provider_call_failed",
                model=model,
                error=error_msg,
                error_type=type(e).__name__,
                traceback=traceback.format_exc()
            )
            raise
        finally:
            self._release_slot(model)

# ==============================================================================
# 10. FACTORY
# ==============================================================================

class ProviderFactory:
    """Create or reuse provider instances based on the model prefix."""
    _instances = {}

    @classmethod
    def get_provider(cls, model_name: str) -> BaseProvider:
        """Return the provider instance responsible for one fully qualified model."""
        prefix = model_name.split("/")[0] if "/" in model_name else "ollama"
        if not is_provider_configured(prefix):
            raise RuntimeError(f"Provider '{prefix}' is not configured in the environment")

        if prefix not in cls._instances:
            if prefix == "openai": cls._instances[prefix] = OpenAIProvider()
            elif prefix == "anthropic": cls._instances[prefix] = AnthropicProvider()
            elif prefix == "gemini": cls._instances[prefix] = GeminiProvider()
            elif prefix == "ollama": cls._instances[prefix] = OllamaProvider()
            else: raise ValueError(f"Namespace desconhecido: {prefix}")
        
        return cls._instances[prefix]

# ==============================================================================
# 11. FUNÇÃO WRAPPER (DROP-IN REPLACEMENT)
# ==============================================================================

async def call_model(
    model: str,
    prompt: str,
    modality: str = "text",
    image_b64: Optional[str] = None,
    temperature: float = 0.5,
    max_tokens: int = 512,
    timeout_seconds: float | None = None,
    workload_class: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run one model call through the normalized provider pipeline.

    This is the main entry point used by the router and judge layers. It
    resolves the provider implementation, executes the async generation path,
    classifies failures into stable application categories, and returns the
    legacy `(text, metadata)` tuple expected by existing callers.
    """
    try:
        if is_provider_temporarily_unavailable(model):
            raise ProviderCallError(
                model=model,
                message="Model temporarily marked unavailable after recent failure",
                category="provider_unavailable",
                retryable=True,
            )
        provider = ProviderFactory.get_provider(model)
        real_name = model.split("/", 1)[1] if "/" in model else model

        result = await provider.generate(
            prompt=prompt,
            image_b64=image_b64,
            model=real_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            workload_class=workload_class,
        )

        meta = _build_response_meta(result)
        clear_provider_unavailable(model)
        return result.text, meta

    except pybreaker.CircuitBreakerError as e:
        raise ProviderCircuitOpenError(model=model, message=str(e)) from e

    except Exception as e:
        structlog_logger.error(
            "call_model_wrapper_failed",
            model=model,
            error=str(e),
            traceback=traceback.format_exc(),
        )
        category = _classify_provider_exception(e)
        if category == "provider_timeout":
            try:
                PROVIDER_TIMEOUT_ERRORS.labels(model=model).inc()
            except Exception:
                pass
        elif category == "provider_rate_limit":
            try:
                PROVIDER_RATE_LIMIT_ERRORS.labels(model=model).inc()
            except Exception:
                pass
        elif category == "provider_unavailable":
            mark_provider_unavailable(model)
        raise ProviderCallError(model=model, message=str(e), category=category, retryable=True) from e


# ==============================================================================
# 12. FUNÇÃO LEGACY (ATUALIZADA: Quick Win #8 - Async com httpx)
# ==============================================================================

async def _ensure_ollama_model_async(model_name: str) -> bool:
    """Ensure an Ollama model is available using the shared async HTTP client."""
    try:
        if is_ollama_model_verified(model_name):
            return True

        existing = [m.get("name") for m in await fetch_ollama_tags(force_refresh=False)]
        if any(model_name in m for m in existing):
            _mark_ollama_models_verified([model_name], 600)
            return True

        existing = [m.get("name") for m in await fetch_ollama_tags(force_refresh=True)]
        if any(model_name in m for m in existing):
            _mark_ollama_models_verified([model_name], 600)
            return True

        logger.info(f"[ensure_ollama_model] Downloading model: {model_name}")
        client = await get_http_client()
        # Use longer timeout for pull (model download can be slow)
        resp = await client.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"name": model_name},
            timeout=900.0  # 15 minutes for large models
        )
        if resp.status_code == 200:
            reset_ollama_tags_cache()
            _mark_ollama_models_verified([model_name], 600)
        return resp.status_code == 200

    except Exception as e:
        logger.warning(f"[ensure_ollama_model] Failed to ensure model {model_name}: {e}")
        return False


def _ensure_ollama_model(model_name: str) -> bool:
    """
    Synchronous function to ensure an Ollama model is available.

    This function is designed to be called from a thread (via asyncio.to_thread).
    It uses synchronous requests library since threads don't have event loops.

    Args:
        model_name: Name of the Ollama model (e.g., "gemma3:4b", "phi4:latest")

    Returns:
        True if model is available or was successfully pulled, False otherwise.
    """
    import requests

    # Defensive: ensure we have an event loop in this thread (Python 3.10+ compatibility)
    # When running in ThreadPoolExecutor, there's no event loop by default.
    # Some libraries or module-level asyncio objects may call get_event_loop(),
    # which raises RuntimeError in Python 3.10+ if no loop exists.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - that's expected for sync code in threads
        # Set a new event loop for this thread to prevent errors from
        # any library that might call get_event_loop()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        if is_ollama_model_verified(model_name):
            return True

        # Check if model already exists
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        if resp.status_code == 200:
            existing = [m.get("name", "") for m in resp.json().get("models", [])]
            # Check for exact match or partial match (e.g., "gemma3:4b" in "gemma3:4b-instruct")
            if any(model_name in m for m in existing):
                _mark_ollama_models_verified(existing, 600)
                logger.debug(f"[ensure_ollama_model] Model {model_name} already exists")
                return True

        # Model not found, attempt to pull
        logger.info(f"[ensure_ollama_model] Downloading model: {model_name}")
        resp = requests.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=900,  # 15 minutes for large models
        )

        if resp.status_code == 200:
            # Consume the streaming response to complete the download
            for _ in resp.iter_lines():
                pass
            reset_ollama_tags_cache()
            _mark_ollama_models_verified([model_name], 600)
            logger.info(f"[ensure_ollama_model] Model {model_name} downloaded successfully")
            return True
        else:
            logger.warning(f"[ensure_ollama_model] Failed to pull model {model_name}: HTTP {resp.status_code}")
            return False

    except requests.exceptions.Timeout:
        logger.warning(f"[ensure_ollama_model] Timeout while ensuring model {model_name}")
        return False
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"[ensure_ollama_model] Connection error for model {model_name}: {e}")
        return False
    except Exception as e:
        logger.warning(f"[ensure_ollama_model] Failed to ensure model {model_name}: {e}")
        return False


def reset_provider_runtime_state() -> None:
    """Reset provider singletons and mutable runtime state for tests and diagnostics."""
    global _http_client
    _http_client = None
    ProviderFactory._instances = {}
    cloud_breaker._state = pybreaker.CircuitClosedState(cloud_breaker)
    local_breaker._state = pybreaker.CircuitClosedState(local_breaker)
    _ollama_concurrency_controller._effective_limit = DEFAULT_OLLAMA_CONCURRENCY_LIMIT
    _ollama_concurrency_controller._last_limit = DEFAULT_OLLAMA_CONCURRENCY_LIMIT
    _ollama_concurrency_controller._latest_vram_used = 0
    _ollama_concurrency_controller._latest_vram_total = 0
    _ollama_concurrency_controller._latest_ratio = 0.0
    _ollama_concurrency_controller._telemetry_ok = False
    _ollama_concurrency_controller._stable_low_windows = 0
    _ollama_runtime_state.clear()
    _provider_unavailable_until.clear()
    reset_ollama_tags_cache()
