# -*- coding: utf-8 -*-
"""
providers_async.py (CORREÇÃO: Suporte a GPT-5/o1 parameter naming)
------------------------------------------------------------------
Implementação Robusta, Assíncrona e Resiliente de TODOS os provedores.

Correções nesta versão:
- Fix: OpenAI gpt-5 e o1 usam 'max_completion_tokens' em vez de 'max_tokens'.
- Resiliência: Circuit Breaker + Tenacity Retry
- Async: HTTPX para Ollama, Threads para Gemini
"""

from __future__ import annotations

import time
import os
import logging
import asyncio
import base64
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List

# Bibliotecas de Resiliência e HTTP Async
import httpx
import pybreaker
import requests # Usado apenas para operações síncronas de setup (ensure model)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# SDKs Defensivos
try:
    from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError
except ImportError:
    AsyncOpenAI = None

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

try:
    import google.generativeai as genai
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
# 1. CIRCUIT BREAKERS GLOBAIS
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
# 2. HELPERS E CONSTANTES
# ==============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

if genai and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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
# 3. MODELOS DE DADOS
# ==============================================================================

class LLMResponse(BaseModel):
    text: str
    latency: float
    cost: float
    prompt_tokens: int
    completion_tokens: int
    model_used: str
    raw_payload: Optional[str] = None 

# ==============================================================================
# 4. ARQUITETURA BASE
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
# 5. PROVEDOR: OPENAI
# ==============================================================================

class OpenAIProvider(BaseProvider):
    def __init__(self):
        if not AsyncOpenAI: raise ImportError("OpenAI SDK not installed")
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        super().__init__("openai", concurrency_limit=100)

    @retry(
        reraise=True, 
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
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

                # --- CORREÇÃO PARA GPT-5 E O1 ---
                # Esses modelos exigem 'max_completion_tokens' e rejeitam 'max_tokens'.
                # Geralmente também não aceitam 'temperature' (ou deve ser fixa).
                if model.startswith("o1-") or "gpt-5" in model:
                    api_args["max_completion_tokens"] = max_tokens
                    # Reasoning models muitas vezes não suportam temperature customizada
                    # Se o gpt-5 suportar, pode descomentar abaixo, mas o default é seguro
                    # api_args["temperature"] = 1.0 
                else:
                    # Modelos legacy (gpt-4, gpt-3.5)
                    api_args["max_tokens"] = max_tokens
                    api_args["temperature"] = temperature

                resp = await self.client.chat.completions.create(**api_args)
                
                text_out = resp.choices[0].message.content or ""
                usage = resp.usage
                p_tok = usage.prompt_tokens if usage else 0
                c_tok = usage.completion_tokens if usage else 0
                
                # Custo via tabela dinâmica
                cost = get_model_cost(model, p_tok, c_tok)
                
                latency = time.time() - start
                self._record_metrics(model, latency, cost, True)
                
                try:
                    raw_payload = json.dumps(resp.model_dump(), default=str)
                except:
                    raw_payload = str(resp)

                return LLMResponse(
                    text=text_out, latency=latency, cost=cost,
                    prompt_tokens=p_tok, completion_tokens=c_tok,
                    model_used=model, raw_payload=raw_payload
                )
            except Exception as e:
                self._record_metrics(model, time.time()-start, 0, False)
                structlog_logger.error("openai_provider_fail", error=str(e), model=model)
                raise

# ==============================================================================
# 6. PROVEDOR: ANTHROPIC
# ==============================================================================

class AnthropicProvider(BaseProvider):
    def __init__(self):
        if not AsyncAnthropic: raise ImportError("Anthropic SDK not installed")
        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        super().__init__("anthropic", concurrency_limit=50)

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
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
                    text=text_out, latency=latency, cost=cost,
                    prompt_tokens=p_tok, completion_tokens=c_tok,
                    model_used=model, raw_payload=str(resp)
                )
            except Exception as e:
                self._record_metrics(model, time.time()-start, 0, False)
                raise

# ==============================================================================
# 7. PROVEDOR: GEMINI (GOOGLE)
# ==============================================================================

class GeminiProvider(BaseProvider):
    def __init__(self):
        if not genai: raise ImportError("Google GenAI SDK not installed")
        super().__init__("gemini", concurrency_limit=60)

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    @cloud_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        model_name = kwargs.get("model", "gemini-1.5-flash")
        start = time.time()

        async with self.semaphore:
            try:
                # Executa chamada síncrona do SDK em thread separada
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
                    text=text_out, latency=latency, cost=cost,
                    prompt_tokens=p_tok, completion_tokens=c_tok,
                    model_used=model_name, raw_payload=str(resp)
                )
            except Exception as e:
                self._record_metrics(model_name, time.time()-start, 0, False)
                raise

# ==============================================================================
# 8. PROVEDOR: OLLAMA (LOCAL & HTTPX ASYNC)
# ==============================================================================

class OllamaProvider(BaseProvider):
    def __init__(self):
        self.host = OLLAMA_HOST
        super().__init__("ollama", concurrency_limit=10) 

    @retry(reraise=True, stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    @local_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        model = kwargs.get("model", "phi4:latest")
        start = time.time()

        async with self.semaphore:
            try:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.5),
                        "num_predict": kwargs.get("max_tokens", 512)
                    }
                }
                if image_b64:
                    payload["images"] = [image_b64]

                async with httpx.AsyncClient(timeout=240.0) as client:
                    resp = await client.post(f"{self.host}/api/generate", json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                text_out = data.get("response", "").strip()
                p_tok = data.get("prompt_eval_count", 0)
                c_tok = data.get("eval_count", 0)
                
                # Custo geralmente zero, mas passa pela tabela para consistência
                cost = get_model_cost(model, p_tok, c_tok)

                latency = time.time() - start
                self._record_metrics(model, latency, cost, True)

                return LLMResponse(
                    text=text_out, latency=latency, cost=cost,
                    prompt_tokens=p_tok, completion_tokens=c_tok,
                    model_used=model, raw_payload=json.dumps(data)
                )
            except Exception as e:
                self._record_metrics(model, time.time()-start, 0, False)
                raise

# ==============================================================================
# 9. FACTORY
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
# 10. FUNÇÃO WRAPPER (DROP-IN REPLACEMENT)
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
            "cost_per_1k": result.cost,
            "quality": heuristic_quality_estimate(result.text),
            "raw_payload": result.raw_payload,
            "image_output_b64": None,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens
        }
        return result.text, meta

    except pybreaker.CircuitBreakerError:
        return "[Erro: Circuito Aberto - Serviço Indisponível]", {"latency": 0.0, "cost_per_1k": 0.0, "quality": 0.0}
    
    except Exception as e:
        structlog_logger.error("provider_call_failed", model=model, error=str(e))
        return f"[Erro ao chamar {model}: {str(e)}]", {"latency": 0.0, "cost_per_1k": 0.0, "quality": 0.0}


# ==============================================================================
# 11. FUNÇÃO LEGACY
# ==============================================================================

def _ensure_ollama_model(model_name: str) -> bool:
    """
    Verifica se o modelo existe no Ollama local; se não, baixa.
    """
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