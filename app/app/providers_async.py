# -*- coding: utf-8 -*-
"""
providers_async.py (FINAL: Suporte a Reasoning/Thinking Explícito)
------------------------------------------------------------------
Implementação Robusta, Assíncrona e Resiliente de TODOS os provedores.

Melhorias:
- Injeção automática de instrução <think> para modelos de raciocínio.
- Extração e separação do pensamento (CoT) da resposta final.
- Timeout estendido para 20 min (1200s) para evitar erros de Cold Start.
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
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
except ImportError:
    genai = None

from pydantic import BaseModel

# Utilitários de Precisão
from app.utils.token_utils import count_tokens
from app.utils.pricing import get_model_cost

# Observabilidade
from app.observability import (
    logger as structlog_logger, 
    registry, 
    PROV_REQ, PROV_ERR, PROV_OK, PROV_LAT, PROV_COST
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from prometheus_client import generate_latest

logger = logging.getLogger("providers_async")

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
# 2. CIRCUIT BREAKERS GLOBAIS
# ==============================================================================

cloud_breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    name="cloud_breaker"
)

local_breaker = pybreaker.CircuitBreaker(
    fail_max=3, 
    reset_timeout=30,
    name="local_breaker"
)

# ==============================================================================
# 3. HELPERS E CONSTANTES
# ==============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

if genai and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Lista de palavras-chave que identificam modelos de raciocínio
REASONING_MODEL_KEYWORDS = ["phi4", "reasoning", "deepseek-r1", "cot", "math"]

def heuristic_quality_estimate(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    length = len(text)
    length_factor = min(1.0, length / 400.0)
    punctuation_bonus = 0.3 if any(p in text for p in [".", "!", "?"]) else 0.0
    score = 4.0 + 5.5 * length_factor + punctuation_bonus
    return max(0.0, min(10.0, round(score, 2)))

def _estimate_tokens(text: str) -> int:
    if not text: return 0
    return max(1, len(text) // 4)

def render_metrics_response():
    return generate_latest(registry), CONTENT_TYPE_LATEST

# ==============================================================================
# 4. MODELOS DE DADOS (ATUALIZADO)
# ==============================================================================

class LLMResponse(BaseModel):
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
    def __init__(self, name: str, concurrency_limit: int):
        self.name = name
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    @abstractmethod
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        pass

    def _record_metrics(self, model: str, latency: float, cost: float, success: bool):
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
    def __init__(self):
        if not AsyncOpenAI: raise ImportError("OpenAI SDK not installed")
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        super().__init__("openai", concurrency_limit=100)

    @COMMON_RETRY_STRATEGY
    @cloud_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
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
                except:
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
    def __init__(self):
        if not AsyncAnthropic: raise ImportError("Anthropic SDK not installed")
        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        super().__init__("anthropic", concurrency_limit=50)

    @COMMON_RETRY_STRATEGY
    @cloud_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
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
    def __init__(self):
        if not genai: raise ImportError("Google GenAI SDK not installed")
        super().__init__("gemini", concurrency_limit=60)

    @COMMON_RETRY_STRATEGY
    @cloud_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        model_name = kwargs.get("model", "gemini-1.5-flash")
        start = time.time()

        async with self.semaphore:
            try:
                def _call():
                    gmodel = genai.GenerativeModel(model_name)
                    parts = [prompt]
                    if image_b64:
                        parts.append({"mime_type": "image/jpeg", "data": base64.b64decode(image_b64)})
                    return gmodel.generate_content(
                        parts,
                        generation_config={
                            "temperature": kwargs.get("temperature", 0.5),
                            "max_output_tokens": kwargs.get("max_tokens", 512)
                        }
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
    def __init__(self):
        self.host = OLLAMA_HOST
        super().__init__("ollama", concurrency_limit=10) 

    @COMMON_RETRY_STRATEGY
    @local_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        model = kwargs.get("model", "phi4:latest")
        start = time.time()

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
                    "options": {
                        "temperature": kwargs.get("temperature", 0.5),
                        "num_predict": kwargs.get("max_tokens", 512),
                        "num_ctx": 4096
                    }
                }
                if image_b64:
                    payload["images"] = [image_b64]

                # Timeout aumentado para 1200s (20 min) para lidar com Cold Start severo em hardware limitado
                async with httpx.AsyncClient(timeout=1200.0) as client:
                    resp = await client.post(f"{self.host}/api/generate", json=payload)
                    resp.raise_for_status() 
                    data = resp.json()

                raw_text = data.get("response", "").strip()
                
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
    _instances = {}

    @classmethod
    def get_provider(cls, model_name: str) -> BaseProvider:
        prefix = model_name.split("/")[0] if "/" in model_name else "ollama"
        
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

        meta = {
            "latency": result.latency,
            "load_time": result.load_time,
            "cost_per_1k": result.cost,
            "quality": heuristic_quality_estimate(result.text),
            "raw_payload": result.raw_payload,
            "image_output_b64": None,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "reasoning": result.reasoning # <--- Passando para cima
        }
        return result.text, meta

    except pybreaker.CircuitBreakerError:
        return "[Erro: Circuito Aberto - Serviço Indisponível]", {"latency": 0.0, "load_time": 0.0, "cost_per_1k": 0.0, "quality": 0.0}
    
    except Exception as e:
        structlog_logger.error(
            "call_model_wrapper_failed", 
            model=model, 
            error=str(e),
            traceback=traceback.format_exc()
        )
        return f"[Erro ao chamar {model}: {str(e)}]", {"latency": 0.0, "load_time": 0.0, "cost_per_1k": 0.0, "quality": 0.0}


# ==============================================================================
# 12. FUNÇÃO LEGACY
# ==============================================================================

def _ensure_ollama_model(model_name: str) -> bool:
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if resp.status_code == 200:
            existing = [m.get("name") for m in resp.json().get("models", [])]
            if any(model_name in m for m in existing):
                return True
        
        logger.info(f"[ensure_ollama_model] Baixando modelo: {model_name}")
        resp = requests.post(f"{OLLAMA_HOST}/api/pull", json={"name": model_name}, stream=True, timeout=900)
        if resp.status_code == 200:
            for _ in resp.iter_lines(): pass
            return True
            
        return False
    except Exception as e:
        logger.warning(f"[ensure_ollama_model] Falha ao garantir modelo {model_name}: {e}")
        return False