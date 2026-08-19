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

import asyncio
import json
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

# Bibliotecas de Resiliência e HTTP Async
import pybreaker
import requests
from prometheus_client import generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from app.model_registry import is_provider_configured
from app.observability import (
    PROVIDER_RATE_LIMIT_ERRORS,
    PROVIDER_TIMEOUT_ERRORS,
    registry,
)

# Observabilidade
from app.observability import (
    logger as structlog_logger,
)
from app.utils.pricing import get_model_cost

# Utilitários de Precisão
from app.utils.token_utils import count_tokens

logger = logging.getLogger("providers_async")


# ============================================================
# Layered submodules (roadmap #19). Public API preserved via reexport below.
# ============================================================
from app.observability import (  # noqa: E402,F401 — reexported for pa.* monkeypatching
    GENERATION_TOKENS_PER_SECOND,
    OLLAMA_DYNAMIC_CONCURRENCY_LIMIT,
    OLLAMA_DYNAMIC_CONCURRENCY_MODE,
    OLLAMA_MODEL_LOAD_SECONDS,
    OLLAMA_MODEL_LOADED,
    PROV_COST,
    PROV_ERR,
    PROV_LAT,
    PROV_OK,
    PROV_REQ,
    PROVIDER_INFLIGHT_REQUESTS,
    PROVIDER_QUEUE_WAIT,
)

from .providers import _implementations, _infra, _ollama  # noqa: E402
from .providers._implementations import (  # noqa: E402,F401
    AnthropicProvider,
    BaseProvider,
    GeminiProvider,
    LLMResponse,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from .providers._infra import (  # noqa: E402,F401  # noqa: E402,F401
    OLLAMA_HOST,
    AsyncAnthropic,
    AsyncOpenAI,
    ProviderCallError,
    ProviderCircuitOpenError,
    _classify_provider_exception,
    _mark_ollama_models_verified,
    _runtime_provider_settings,
    fetch_ollama_tags,
    genai,
    get_http_client,
    is_ollama_model_verified,
    reset_ollama_tags_cache,
)
from .providers._ollama import (  # noqa: E402,F401
    _get_adaptive_timeout,  # noqa: E402,F401
    clear_provider_unavailable,
    is_provider_temporarily_unavailable,
    mark_provider_unavailable,
)


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
        "tool_calls": result.tool_calls,
        "finish_reason": result.finish_reason,
    }


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
    if not text:
        return 0
    return max(1, len(text) // 4)


def render_metrics_response():
    """Serialize provider metrics using the shared Prometheus registry."""
    return generate_latest(registry), CONTENT_TYPE_LATEST


# ==============================================================================
# 4. MODELOS DE DADOS (ATUALIZADO)
# ==============================================================================


# ==============================================================================
# 10. FACTORY
# ==============================================================================


class ProviderFactory:
    """Create or reuse provider instances based on the model prefix."""

    _instances: dict[str, BaseProvider] = {}

    @classmethod
    def get_provider(cls, model_name: str) -> BaseProvider:
        """Return the provider instance responsible for one fully qualified model."""
        prefix = model_name.split("/")[0] if "/" in model_name else "ollama"
        if not is_provider_configured(prefix):
            raise RuntimeError(f"Provider '{prefix}' is not configured in the environment")

        if prefix not in cls._instances:
            if prefix == "openai":
                cls._instances[prefix] = OpenAIProvider()
            elif prefix == "openrouter":
                cls._instances[prefix] = OpenRouterProvider()
            elif prefix == "anthropic":
                cls._instances[prefix] = AnthropicProvider()
            elif prefix == "gemini":
                cls._instances[prefix] = GeminiProvider()
            elif prefix == "ollama":
                cls._instances[prefix] = OllamaProvider()
            else:
                raise ValueError(f"Namespace desconhecido: {prefix}")

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
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
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
            tools=tools,
            tool_choice=tool_choice,
            messages=messages,
            response_format=response_format,
            system_prompt=system_prompt,
        )

        # Turno de tool sem texto: se o provider não contabilizou os tokens de
        # completion, estima a partir dos tool_calls para o custo não zerar.
        if result.tool_calls and not result.completion_tokens:
            try:
                result.completion_tokens = count_tokens(json.dumps(result.tool_calls, ensure_ascii=False), real_name)
                result.cost = get_model_cost(model, result.prompt_tokens, result.completion_tokens)
            except Exception:
                pass

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
        if any(model_name in (m or "") for m in existing):
            _mark_ollama_models_verified([model_name], 600)
            return True

        existing = [m.get("name") for m in await fetch_ollama_tags(force_refresh=True)]
        if any(model_name in (m or "") for m in existing):
            _mark_ollama_models_verified([model_name], 600)
            return True

        logger.info(f"[ensure_ollama_model] Downloading model: {model_name}")
        client = await get_http_client()
        # Use longer timeout for pull (model download can be slow)
        resp = await client.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"name": model_name},
            timeout=900.0,  # 15 minutes for large models
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
    _infra.reset_state()
    _ollama.reset_state()
    ProviderFactory._instances = {}


def __getattr__(name):  # PEP 562: reexport any remaining submodule symbol
    for _mod in (_infra, _ollama, _implementations):
        if hasattr(_mod, name):
            return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
