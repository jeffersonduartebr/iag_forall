# -*- coding: utf-8 -*-
# Objective: Shared provider base: normalized response, abstract provider and facade shims.
"""LLMResponse, BaseProvider and the shims that route test-patched hot symbols
(get_http_client, get_model_cost, count_tokens, _runtime_provider_settings)
through ``app.providers_async`` so ``pa.*`` monkeypatches keep working."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

import app.providers_async as _pa
from app.services.orcamento_tempo import registrar_vazao
from app.services.sombra.contexto import em_sombra

from ._ollama import (
    _mark_ollama_model_state,
)

logger = _pa.logger if hasattr(_pa, "logger") else __import__("logging").getLogger("providers_async")


logger = _pa.logger if hasattr(_pa, "logger") else __import__("logging").getLogger("providers_async")


def get_model_cost(*args, **kwargs):
    return _pa.get_model_cost(*args, **kwargs)


def count_tokens(*args, **kwargs):
    return _pa.count_tokens(*args, **kwargs)


async def get_http_client():
    return await _pa.get_http_client()


def _runtime_provider_settings():
    return _pa._runtime_provider_settings()


class LLMResponse(BaseModel):
    """Represent the normalized provider response consumed by router code.

    Every provider adapter returns this model so downstream code can work with a
    stable shape for text, timing, cost, token counts, raw payloads, and
    optional reasoning traces.
    """

    text: str
    latency: float
    load_time: float = 0.0
    cost: float  # custo de caixa (cobrança do provedor)
    cost_imputed: float = 0.0  # custo imputado da ocupação do equipamento local
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int = 0  # parte de completion_tokens gasta em raciocínio, quando o provedor informa
    model_used: str
    raw_payload: Optional[str] = None
    reasoning: Optional[str] = None  # <--- NOVO CAMPO: Armazena o pensamento (CoT)
    tool_calls: Optional[List[Dict[str, Any]]] = None  # Tool/function calls no formato canônico (OpenAI)
    finish_reason: Optional[str] = None  # "stop" | "tool_calls" | "length" | ...


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
        if em_sombra():  # a ocupação e as métricas de fila contam só o atendimento real (R2, R10)
            return
        if self.name == "ollama":
            _mark_ollama_model_state(model, inflight_delta=1, queue_wait_seconds=waited)
        try:
            _pa.PROVIDER_QUEUE_WAIT.labels(model=model).observe(waited)
            _pa.PROVIDER_INFLIGHT_REQUESTS.labels(model=model).inc()
        except Exception:
            pass

    def _release_slot(self, model: str) -> None:
        """Release provider capacity and publish in-flight metrics."""
        try:
            self.semaphore.release()
        finally:
            if em_sombra():
                return
            if self.name == "ollama":
                _mark_ollama_model_state(model, inflight_delta=-1)
            try:
                _pa.PROVIDER_INFLIGHT_REQUESTS.labels(model=model).dec()
            except Exception:
                pass

    def _record_metrics(self, model: str, latency: float, cost: float, success: bool):
        """Publish provider-level success, latency, and cost metrics (never for shadow calls)."""
        if em_sombra():
            return
        _pa.PROV_REQ.labels(model=model).inc()
        if success:
            _pa.PROV_OK.labels(model=model).inc()
            _pa.PROV_LAT.labels(model=model).observe(latency)
            _pa.PROV_COST.labels(model=model).observe(cost)
        else:
            _pa.PROV_ERR.labels(model=model).inc()

    def _record_generation_metrics(self, model: str, completion_tokens: int, latency: float) -> None:
        """Publish generation throughput metrics and feed the throughput EMA (never for shadow calls)."""
        if em_sombra():
            return
        if latency <= 0 or completion_tokens <= 0:
            return
        try:
            _pa.GENERATION_TOKENS_PER_SECOND.labels(model=model).observe(completion_tokens / latency)
        except Exception:
            pass
        registrar_vazao(model, completion_tokens, latency)
