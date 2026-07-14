# Objective: Provider infra — errors, retry, breakers, HTTP pool, config (roadmap #19).
"""Foundational provider layer extracted from providers_async: exceptions, retry
strategy, circuit breakers, the shared HTTP client + Ollama tag cache, and the
runtime provider settings/constants. Leaf module (no provider-internal imports)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import resource
import threading
import time
import warnings
from typing import Any, Dict, List, cast

import httpx
import pybreaker
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

# SDKs Defensivos
try:
    from openai import APIConnectionError, APITimeoutError, AsyncOpenAI
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]

try:
    from anthropic import APIStatusError, AsyncAnthropic
    from anthropic import RateLimitError as AnthropicRateLimitError
except ImportError:
    AsyncAnthropic = None  # type: ignore[assignment,misc]

try:
    from google import genai as google_genai  # type: ignore[attr-defined]
except ImportError:
    google_genai = None

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
except ImportError:
    genai = None  # type: ignore[assignment]

import app.providers_async as _pa

logger = logging.getLogger("providers_async")


def reset_state() -> None:
    """Reset infra-owned mutable state (HTTP client, breakers, verified models)."""
    _pa._http_client = None
    reset_ollama_tags_cache()
    cloud_breaker._state = pybreaker.CircuitClosedState(cloud_breaker)
    local_breaker._state = pybreaker.CircuitClosedState(local_breaker)
    with _ollama_verified_models_lock:
        _ollama_verified_models.clear()


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


# ==============================================================================
# 1. ESTRATÉGIA DE RETRY (RATE LIMITS)
# ==============================================================================

RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.HTTPStatusError,
)

if AsyncOpenAI is not None:
    RETRYABLE_ERRORS += (APITimeoutError, APIConnectionError, OpenAIRateLimitError)

if AsyncAnthropic is not None:
    RETRYABLE_ERRORS += (AnthropicRateLimitError, APIStatusError)

if genai:
    RETRYABLE_ERRORS += (ResourceExhausted, ServiceUnavailable)

COMMON_RETRY_STRATEGY = retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
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

cloud_breaker = pybreaker.CircuitBreaker(fail_max=CB_FAIL_MAX, reset_timeout=CB_RESET_TIMEOUT, name="cloud_breaker")

local_breaker = pybreaker.CircuitBreaker(
    fail_max=CB_LOCAL_FAIL_MAX, reset_timeout=CB_LOCAL_RESET_TIMEOUT, name="local_breaker"
)

# ==============================================================================
# 3. HTTP CONNECTION POOLING (Performance Optimization)
# ==============================================================================

# Global HTTP client for connection reuse
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
        return any(
            _ollama_verified_models.get(alias, 0.0) > now for alias in _normalize_ollama_model_aliases(model_name)
        )


async def get_http_client() -> httpx.AsyncClient:
    """
    Get or create the shared async HTTP client used by provider integrations.

    The client is initialized lazily and reused across calls to avoid repeated
    connection setup overhead. A shared pool matters especially for Ollama and
    cloud SDK fallbacks that issue frequent HTTP requests.
    """

    if _pa._http_client is None or _pa._http_client.is_closed:
        async with _http_client_lock:
            # Double-check after acquiring lock
            if _pa._http_client is None or _pa._http_client.is_closed:
                _pa._http_client = httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=200,  # Optimized for high-capacity environment
                        max_keepalive_connections=50,
                    ),
                    timeout=httpx.Timeout(60.0, connect=10.0),
                    http2=True,  # Enable HTTP/2 for better performance
                )
                logger.info("[HTTP Pool] Created global HTTP client with connection pooling (200/50)")

    return _pa._http_client


async def close_http_client():
    """Close the shared HTTP client during process shutdown."""
    if _pa._http_client is not None and not _pa._http_client.is_closed:
        await _pa._http_client.aclose()
        _pa._http_client = None
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

        client = await _pa.get_http_client()
        resp = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=10.0)
        resp.raise_for_status()
        models = list(resp.json().get("models", []))
        _ollama_tags_cache["models"] = models
        _ollama_tags_cache["expires_at"] = time.time() + max(1, ttl_seconds)
        _mark_ollama_models_verified([m.get("name", "") for m in models], ttl_seconds)
        return list(models)


async def probe_ollama_liveness() -> None:
    """Perform a lightweight Ollama liveness probe using the shared HTTP pool."""
    client = await _pa.get_http_client()
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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "").strip()
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

        out["ollama_concurrency_limit"] = max(
            1, int(dynamic_settings.get("OLLAMA_CONCURRENCY_LIMIT", str(DEFAULT_OLLAMA_CONCURRENCY_LIMIT)))
        )
        out["ollama_dynamic_concurrency_enabled"] = str(
            dynamic_settings.get("OLLAMA_DYNAMIC_CONCURRENCY_ENABLED", "0")
        ).strip() in ("1", "true", "True")
        out["ollama_concurrency_min"] = max(
            1, int(dynamic_settings.get("OLLAMA_CONCURRENCY_MIN", str(DEFAULT_OLLAMA_CONCURRENCY_MIN)))
        )
        out["ollama_concurrency_max"] = max(
            cast(int, out["ollama_concurrency_min"]),
            int(dynamic_settings.get("OLLAMA_CONCURRENCY_MAX", str(out["ollama_concurrency_limit"]))),
        )
        out["ollama_vram_target_utilization"] = min(
            0.99,
            max(
                0.1,
                float(
                    dynamic_settings.get("OLLAMA_VRAM_TARGET_UTILIZATION", str(DEFAULT_OLLAMA_VRAM_TARGET_UTILIZATION))
                ),
            ),
        )
        out["ollama_vram_high_watermark"] = min(
            0.99,
            max(
                0.1, float(dynamic_settings.get("OLLAMA_VRAM_HIGH_WATERMARK", str(DEFAULT_OLLAMA_VRAM_HIGH_WATERMARK)))
            ),
        )
        out["ollama_vram_low_watermark"] = min(
            cast(float, out["ollama_vram_high_watermark"]),
            max(0.0, float(dynamic_settings.get("OLLAMA_VRAM_LOW_WATERMARK", str(DEFAULT_OLLAMA_VRAM_LOW_WATERMARK)))),
        )
        out["ollama_vram_poll_interval_seconds"] = max(
            1,
            int(
                dynamic_settings.get(
                    "OLLAMA_VRAM_POLL_INTERVAL_SECONDS", str(DEFAULT_OLLAMA_VRAM_POLL_INTERVAL_SECONDS)
                )
            ),
        )
        out["ollama_concurrency_step_up"] = max(
            1, int(dynamic_settings.get("OLLAMA_CONCURRENCY_STEP_UP", str(DEFAULT_OLLAMA_CONCURRENCY_STEP_UP)))
        )
        out["ollama_concurrency_step_down"] = max(
            1, int(dynamic_settings.get("OLLAMA_CONCURRENCY_STEP_DOWN", str(DEFAULT_OLLAMA_CONCURRENCY_STEP_DOWN)))
        )
        out["ollama_concurrency_stable_windows"] = max(
            1,
            int(
                dynamic_settings.get(
                    "OLLAMA_CONCURRENCY_STABLE_WINDOWS", str(DEFAULT_OLLAMA_CONCURRENCY_STABLE_WINDOWS)
                )
            ),
        )
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
                parsed_interactive_warm_models = [
                    item.strip() for item in interactive_warm_models_raw.split(",") if item.strip()
                ]
        elif isinstance(interactive_warm_models_raw, (list, tuple, set)):
            parsed_interactive_warm_models = list(interactive_warm_models_raw)
        else:
            parsed_interactive_warm_models = []
        out["ollama_interactive_warm_models"] = [
            (str(model).strip() if str(model).strip().startswith("ollama/") else f"ollama/{str(model).strip()}")
            for model in parsed_interactive_warm_models
            if str(model).strip()
        ]
        out["ollama_warmup_generate_enabled"] = str(
            dynamic_settings.get("OLLAMA_WARMUP_GENERATE_ENABLED", "1")
        ).strip() in ("1", "true", "True")
        out["ollama_load_penalty_enabled"] = str(dynamic_settings.get("OLLAMA_LOAD_PENALTY_ENABLED", "1")).strip() in (
            "1",
            "true",
            "True",
        )
        out["ollama_background_load_shedding_enabled"] = str(
            dynamic_settings.get("OLLAMA_BACKGROUND_LOAD_SHEDDING_ENABLED", "1")
        ).strip() in ("1", "true", "True")
        out["ollama_route_candidate_limit"] = max(
            2, int(dynamic_settings.get("OLLAMA_ROUTE_CANDIDATE_LIMIT", str(DEFAULT_OLLAMA_ROUTE_CANDIDATE_LIMIT)))
        )
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
        out["background_throttle_inflight_threshold"] = max(
            1,
            int(
                dynamic_settings.get(
                    "BACKGROUND_THROTTLE_INFLIGHT_THRESHOLD", str(DEFAULT_BACKGROUND_THROTTLE_INFLIGHT_THRESHOLD)
                )
            ),
        )
        out["background_throttle_queue_wait_p95_ms"] = max(
            1,
            int(
                dynamic_settings.get(
                    "BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS", str(DEFAULT_BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS)
                )
            ),
        )
        out["adaptive_timeout_enabled"] = str(dynamic_settings.get("ADAPTIVE_TIMEOUT_ENABLED", "1")).strip() in (
            "1",
            "true",
            "True",
        )
        out["base_timeout"] = max(1, int(dynamic_settings.get("MIN_TIMEOUT", str(DEFAULT_BASE_TIMEOUT))))
        out["max_timeout"] = max(
            cast(int, out["base_timeout"]), int(dynamic_settings.get("MAX_TIMEOUT", str(DEFAULT_MAX_TIMEOUT)))
        )
        out["timeout_multiplier"] = max(
            1.0, float(dynamic_settings.get("ADAPTIVE_TIMEOUT_MULTIPLIER", str(DEFAULT_TIMEOUT_MULTIPLIER)))
        )
        out["reasoning_multiplier"] = max(
            1.0, float(dynamic_settings.get("ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER", str(DEFAULT_REASONING_MULTIPLIER)))
        )
    except Exception:
        pass
    return out
