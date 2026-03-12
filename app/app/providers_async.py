# -*- coding: utf-8 -*-
"""
providers_async.py (FINAL: Suporte a Reasoning/Thinking Explícito)
------------------------------------------------------------------
Implementação Robusta, Assíncrona e Resiliente de TODOS os provedores.

Melhorias:
- Injeção automática de instrução <think> para modelos de raciocínio.
- Extração e separação do pensamento (CoT) da resposta final.
- Quick Win #6: Adaptive timeout based on model type.
- Quick Win #7: Configurable Ollama concurrency limit.
- Quick Win #8: Async Ollama model ensure using httpx.
"""

from __future__ import annotations

import time
import os
import logging
import asyncio
import base64
import json
import traceback
import re
import warnings
from types import SimpleNamespace
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple

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
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from prometheus_client import generate_latest

logger = logging.getLogger("providers_async")

# ==============================================================================
# Provider Exceptions
# ==============================================================================


class ProviderCallError(Exception):
    """Structured provider call failure."""

    def __init__(
        self,
        model: str,
        message: str,
        *,
        category: str = "provider_unavailable",
        retryable: bool = True,
    ):
        """Inicializa estado interno necessário para uso da classe."""
        super().__init__(message)
        self.model = model
        self.category = category
        self.retryable = retryable


class ProviderCircuitOpenError(ProviderCallError):
    """Raised when a provider circuit breaker is open."""

    def __init__(self, model: str, message: str):
        """Inicializa estado interno necessário para uso da classe."""
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
    """Build a consistent metadata payload from LLMResponse."""
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


async def get_http_client() -> httpx.AsyncClient:
    """
    Get or create a global HTTP client with connection pooling.

    Uses lazy initialization with async lock for thread safety.
    Connection limits:
    - max_connections: 100 (total connections across all hosts)
    - max_keepalive_connections: 20 (persistent connections per host)
    - Timeouts: 60s total, 10s for connect
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
    """Close the global HTTP client. Call on shutdown."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.close()
        _http_client = None
        logger.info("[HTTP Pool] Closed global HTTP client")


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
DEFAULT_ADAPTIVE_TIMEOUT_ENABLED = True
DEFAULT_BASE_TIMEOUT = 60
DEFAULT_MAX_TIMEOUT = 1200
DEFAULT_TIMEOUT_MULTIPLIER = 2.0
DEFAULT_REASONING_MULTIPLIER = 3.0

# Backward-compatible exported values (used by tests/legacy code)
OLLAMA_CONCURRENCY_LIMIT = DEFAULT_OLLAMA_CONCURRENCY_LIMIT
ADAPTIVE_TIMEOUT_ENABLED = DEFAULT_ADAPTIVE_TIMEOUT_ENABLED
BASE_TIMEOUT = DEFAULT_BASE_TIMEOUT
MAX_TIMEOUT = DEFAULT_MAX_TIMEOUT
TIMEOUT_MULTIPLIER = DEFAULT_TIMEOUT_MULTIPLIER
REASONING_MULTIPLIER = DEFAULT_REASONING_MULTIPLIER


def _runtime_provider_settings() -> Dict[str, Any]:
    """Read provider runtime settings with safe fallback (supports hot reload)."""
    out = {
        "ollama_concurrency_limit": DEFAULT_OLLAMA_CONCURRENCY_LIMIT,
        "adaptive_timeout_enabled": DEFAULT_ADAPTIVE_TIMEOUT_ENABLED,
        "base_timeout": DEFAULT_BASE_TIMEOUT,
        "max_timeout": DEFAULT_MAX_TIMEOUT,
        "timeout_multiplier": DEFAULT_TIMEOUT_MULTIPLIER,
        "reasoning_multiplier": DEFAULT_REASONING_MULTIPLIER,
    }
    try:
        from app.settings_dynamic import settings as dynamic_settings
        out["ollama_concurrency_limit"] = max(1, int(dynamic_settings.get("OLLAMA_CONCURRENCY_LIMIT", str(DEFAULT_OLLAMA_CONCURRENCY_LIMIT))))
        out["adaptive_timeout_enabled"] = str(dynamic_settings.get("ADAPTIVE_TIMEOUT_ENABLED", "1")).strip() in ("1", "true", "True")
        out["base_timeout"] = max(1, int(dynamic_settings.get("MIN_TIMEOUT", str(DEFAULT_BASE_TIMEOUT))))
        out["max_timeout"] = max(out["base_timeout"], int(dynamic_settings.get("MAX_TIMEOUT", str(DEFAULT_MAX_TIMEOUT))))
        out["timeout_multiplier"] = max(1.0, float(dynamic_settings.get("ADAPTIVE_TIMEOUT_MULTIPLIER", str(DEFAULT_TIMEOUT_MULTIPLIER))))
        out["reasoning_multiplier"] = max(1.0, float(dynamic_settings.get("ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER", str(DEFAULT_REASONING_MULTIPLIER))))
    except Exception:
        pass
    return out


def _get_adaptive_timeout(model: str) -> float:
    """
    Calculate adaptive timeout based on model type (Quick Win #6).
    - Reasoning models: 3x base timeout
    - Large models: 2x base timeout
    - Standard models: base timeout
    """
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
    """Executa heuristic quality estimate."""
    if not text or not text.strip():
        return 0.0
    length = len(text)
    length_factor = min(1.0, length / 400.0)
    punctuation_bonus = 0.3 if any(p in text for p in [".", "!", "?"]) else 0.0
    score = 4.0 + 5.5 * length_factor + punctuation_bonus
    return max(0.0, min(10.0, round(score, 2)))

def _estimate_tokens(text: str) -> int:
    """Executa estimate tokens."""
    if not text: return 0
    return max(1, len(text) // 4)

def render_metrics_response():
    """Executa render metrics response."""
    return generate_latest(registry), CONTENT_TYPE_LATEST

# ==============================================================================
# 4. MODELOS DE DADOS (ATUALIZADO)
# ==============================================================================

class LLMResponse(BaseModel):
    """Classe `LLMResponse`: organiza responsabilidades de providers async."""
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
    """Classe `BaseProvider`: organiza responsabilidades de providers async."""
    def __init__(self, name: str, concurrency_limit: int):
        """Inicializa estado interno necessário para uso da classe."""
        self.name = name
        self._concurrency_limit = max(1, int(concurrency_limit))
        self.semaphore = asyncio.Semaphore(self._concurrency_limit)

    @abstractmethod
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        """Executa generate."""
        pass

    def _record_metrics(self, model: str, latency: float, cost: float, success: bool):
        """Executa record metrics."""
        PROV_REQ.labels(model=model).inc()
        if success:
            PROV_OK.labels(model=model).inc()
            PROV_LAT.labels(model=model).observe(latency)
            PROV_COST.labels(model=model).observe(cost)
        else:
            PROV_ERR.labels(model=model).inc()

# ==============================================================================
# 6. PROVEDOR: OPENAI
# ==============================================================================

class OpenAIProvider(BaseProvider):
    """Classe `OpenAIProvider`: organiza responsabilidades de providers async."""
    def __init__(self):
        """Inicializa estado interno necessário para uso da classe."""
        if not AsyncOpenAI: raise ImportError("OpenAI SDK not installed")
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        super().__init__("openai", concurrency_limit=100)

    @COMMON_RETRY_STRATEGY
    @cloud_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        """Executa generate."""
        model = kwargs.get("model", "gpt-4o")
        temperature = kwargs.get("temperature", 0.5)
        max_tokens = kwargs.get("max_tokens", 512)
        start = time.time()

        async with self.semaphore:
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
                
                try:
                    raw_payload = json.dumps(resp.model_dump(), default=str)
                except Exception:
                    raw_payload = str(resp)

                return LLMResponse(
                    text=text_out, latency=latency, load_time=0.0,
                    cost=cost, prompt_tokens=p_tok, completion_tokens=c_tok,
                    model_used=model, raw_payload=raw_payload,
                    reasoning=None # OpenAI o1 não expõe o reasoning token a token publicamente ainda
                )
            except Exception as e:
                self._record_metrics(model, time.time()-start, 0, False)
                structlog_logger.error("openai_provider_fail", error=str(e), model=model)
                raise

# ==============================================================================
# 7. PROVEDOR: ANTHROPIC
# ==============================================================================

class AnthropicProvider(BaseProvider):
    """Classe `AnthropicProvider`: organiza responsabilidades de providers async."""
    def __init__(self):
        """Inicializa estado interno necessário para uso da classe."""
        if not AsyncAnthropic: raise ImportError("Anthropic SDK not installed")
        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        super().__init__("anthropic", concurrency_limit=50)

    @COMMON_RETRY_STRATEGY
    @cloud_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        """Executa generate."""
        model = kwargs.get("model", "claude-3-5-sonnet-latest")
        start = time.time()

        async with self.semaphore:
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

                return LLMResponse(
                    text=text_out, latency=latency, load_time=0.0,
                    cost=cost, prompt_tokens=p_tok, completion_tokens=c_tok,
                    model_used=model, raw_payload=str(resp),
                    reasoning=None
                )
            except Exception as e:
                self._record_metrics(model, time.time()-start, 0, False)
                raise

# ==============================================================================
# 8. PROVEDOR: GEMINI (GOOGLE)
# ==============================================================================

class GeminiProvider(BaseProvider):
    """Classe `GeminiProvider`: organiza responsabilidades de providers async."""
    class GeminiAdapter:
        """Adapter layer to isolate SDK calls and simplify migration to google.genai."""

        def generate(
            self,
            model_name: str,
            prompt: str,
            image_b64: Optional[str],
            temperature: float,
            max_tokens: int,
        ):
            # Prefer google.genai (new SDK), fallback to google.generativeai.
            """Executa generate."""
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
        """Inicializa estado interno necessário para uso da classe."""
        if not genai: raise ImportError("Google GenAI SDK not installed")
        super().__init__("gemini", concurrency_limit=60)
        self._adapter = self.GeminiAdapter()

    @COMMON_RETRY_STRATEGY
    @cloud_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        """Executa generate."""
        model_name = kwargs.get("model", "gemini-1.5-flash")
        start = time.time()

        async with self.semaphore:
            try:
                def _call():
                    """Executa a responsabilidade descrita por este método.

                    Returns:
                        Valor produzido pela execução.
                    """
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

                return LLMResponse(
                    text=text_out, latency=latency, load_time=0.0,
                    cost=cost, prompt_tokens=p_tok, completion_tokens=c_tok,
                    model_used=model_name, raw_payload=str(resp),
                    reasoning=None
                )
            except Exception as e:
                self._record_metrics(model_name, time.time()-start, 0, False)
                raise

# ==============================================================================
# 9. PROVEDOR: OLLAMA (LOCAL & HTTPX ASYNC)
# ==============================================================================

class OllamaProvider(BaseProvider):
    """Classe `OllamaProvider`: organiza responsabilidades de providers async."""
    def __init__(self):
        """Inicializa estado interno necessário para uso da classe."""
        self.host = OLLAMA_HOST
        cfg = _runtime_provider_settings()
        super().__init__("ollama", concurrency_limit=int(cfg["ollama_concurrency_limit"]))

    def _refresh_concurrency_limit(self) -> None:
        """Executa refresh concurrency limit."""
        cfg = _runtime_provider_settings()
        new_limit = int(cfg["ollama_concurrency_limit"])
        if new_limit != self._concurrency_limit:
            self._concurrency_limit = new_limit
            self.semaphore = asyncio.Semaphore(new_limit)
            logger.info("[ollama] Updated concurrency limit to %s", new_limit)

    @COMMON_RETRY_STRATEGY
    @local_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        """Executa generate."""
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

        async with self.semaphore:
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
                timeout = _get_adaptive_timeout(model)

                # Use pooled HTTP client for better performance
                client = await get_http_client()
                resp = await client.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=timeout  # Override default timeout for this request
                )
                resp.raise_for_status()
                data = resp.json()

                raw_text = data.get("response", "").strip()
                raw_thinking = data.get("thinking", "").strip()
                
                # --- EXTRAÇÃO DE REASONING (THINKING) ---
                # Modelos como Phi-4 e DeepSeek-R1 retornam o raciocínio dentro de <think>...</think>
                reasoning = None
                text_out = raw_text

                # Regex para capturar o conteúdo dentro de <think>
                think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
                if think_match:
                    reasoning = think_match.group(1).strip()
                    # Remove o pensamento da resposta final para o usuário
                    text_out = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                elif raw_thinking:
                    reasoning = raw_thinking
                
                # --- EXTRAÇÃO DE LOAD TIME ---
                load_ns = data.get("load_duration", 0)
                load_sec = float(load_ns) / 1_000_000_000.0
                
                p_tok = data.get("prompt_eval_count", 0)
                c_tok = data.get("eval_count", 0)
                cost = get_model_cost(model, p_tok, c_tok)
                latency = time.time() - start
                
                self._record_metrics(model, latency, cost, True)

                return LLMResponse(
                    text=text_out, 
                    latency=latency, 
                    load_time=load_sec,
                    cost=cost,
                    prompt_tokens=p_tok, 
                    completion_tokens=c_tok,
                    model_used=model, 
                    raw_payload=json.dumps(data),
                    reasoning=reasoning # <--- Campo preenchido se houver <think>
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

# ==============================================================================
# 10. FACTORY
# ==============================================================================

class ProviderFactory:
    """Classe `ProviderFactory`: organiza responsabilidades de providers async."""
    _instances = {}

    @classmethod
    def get_provider(cls, model_name: str) -> BaseProvider:
        """Obtém provider."""
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
) -> Tuple[str, Dict[str, Any]]:
    """
    Substituição direta (drop-in) para a antiga função call_model.
    Roteia internamente para a arquitetura async/resiliente.
    """
    try:
        provider = ProviderFactory.get_provider(model)
        real_name = model.split("/", 1)[1] if "/" in model else model

        result = await provider.generate(
            prompt=prompt,
            image_b64=image_b64,
            model=real_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        meta = _build_response_meta(result)
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
        raise ProviderCallError(model=model, message=str(e), category=category, retryable=True) from e


# ==============================================================================
# 12. FUNÇÃO LEGACY (ATUALIZADA: Quick Win #8 - Async com httpx)
# ==============================================================================

async def _ensure_ollama_model_async(model_name: str) -> bool:
    """Async version using pooled httpx client (Quick Win #8)."""
    try:
        client = await get_http_client()
        resp = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=10.0)
        if resp.status_code == 200:
            existing = [m.get("name") for m in resp.json().get("models", [])]
            if any(model_name in m for m in existing):
                return True

        logger.info(f"[ensure_ollama_model] Downloading model: {model_name}")
        # Use longer timeout for pull (model download can be slow)
        resp = await client.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"name": model_name},
            timeout=900.0  # 15 minutes for large models
        )
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
        # Check if model already exists
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        if resp.status_code == 200:
            existing = [m.get("name", "") for m in resp.json().get("models", [])]
            # Check for exact match or partial match (e.g., "gemma3:4b" in "gemma3:4b-instruct")
            if any(model_name in m for m in existing):
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
    """
    Reset global/singleton provider runtime state (test/dev utility).
    """
    global _http_client
    _http_client = None
    ProviderFactory._instances = {}
    cloud_breaker._state = pybreaker.CircuitClosedState(cloud_breaker)
    local_breaker._state = pybreaker.CircuitClosedState(local_breaker)
