# -*- coding: utf-8 -*-
# Objective: ollama_adapter provider adapter (split from providers/_implementations.py).
"""Concrete provider adapter; shared base and shims live in ``providers._base``."""

from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import httpx

import app.providers_async as _pa
from app import provider_tools as ptools  # type: ignore[attr-defined]
from app.observability import logger as structlog_logger
from app.utils.breaker_async import guarded_by
from app.utils.pricing import impute_local_cost

from ._base import (
    BaseProvider,
    LLMResponse,
    _runtime_provider_settings,
    get_http_client,
    get_model_cost,
    logger,
)
from ._infra import (
    COMMON_RETRY_STRATEGY,
    OLLAMA_HOST,
    REASONING_MODEL_KEYWORDS,
    local_breaker,
)
from ._ollama import (
    _mark_ollama_model_state,
    _ollama_concurrency_controller,
)


def _split_reasoning(raw_text: str, raw_thinking: str) -> Tuple[str, Optional[str]]:
    """Separate an inline ``<think>`` block from the answer.

    Some models emit the reasoning inside the response text, others return it
    on Ollama's own ``thinking`` field. The inline block wins when both exist,
    because it is the one that would otherwise leak into the answer.
    """
    match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
    if match:
        return re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip(), match.group(1).strip()
    return raw_text, raw_thinking or None


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

    async def _chat_call(self, client, *, model, prompt, image_b64, options, timeout, ollama_format, is_reasoning, kwargs):
        """POST ``/api/chat`` — the multi-turn and tool-calling path.

        ``is_reasoning`` is accepted and ignored: this endpoint has no separate
        thinking channel, and taking the same arguments as
        :meth:`_generate_call` is what lets the caller pick one without a
        branch around every parameter.
        """
        canonical_messages = ptools.build_provider_messages(
            prompt, kwargs.get("system_prompt"), kwargs.get("messages"), image_b64
        )
        chat_payload: Dict[str, Any] = {
            "model": model,
            "messages": ptools.to_ollama_messages(canonical_messages),
            "stream": False,
            "options": options,
        }
        tools = kwargs.get("tools")
        if tools and not ptools.tools_disabled(kwargs.get("tool_choice")):
            chat_payload["tools"] = tools
        if ollama_format is not None:
            chat_payload["format"] = ollama_format
        resp = await client.post(f"{self.host}/api/chat", json=chat_payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text_out, tool_calls, finish_reason = ptools.from_ollama_chat(data)
        return data, text_out, tool_calls, finish_reason, None

    async def _generate_call(self, client, *, model, prompt, image_b64, options, timeout, ollama_format, is_reasoning, kwargs):
        """POST ``/api/generate`` — the single-turn path, with a reasoning channel."""
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            # Avoid empty final answers on models that support a separate
            # thinking channel (for example qwen3.5) unless we explicitly
            # want reasoning output.
            "think": is_reasoning,
            "options": options,
        }
        if image_b64:
            payload["images"] = [image_b64]
        if ollama_format is not None:
            payload["format"] = ollama_format
        resp = await client.post(f"{self.host}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text_out, reasoning = _split_reasoning(
            data.get("response", "").strip(), data.get("thinking", "").strip()
        )
        return data, text_out, None, "stop", reasoning

    def _record_load(self, model: str, load_sec: float) -> None:
        """Model-load telemetry. A metrics backend must never break a response."""
        if load_sec > 0:
            try:
                _pa.OLLAMA_MODEL_LOAD_SECONDS.labels(model=model).observe(load_sec)
            except Exception:
                pass
        try:
            _pa.OLLAMA_MODEL_LOADED.labels(model=model).set(1)
        except Exception:
            pass
        _mark_ollama_model_state(model, loaded=True, load_seconds=load_sec)

    @guarded_by(local_breaker)
    @COMMON_RETRY_STRATEGY
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        """Execute one local Ollama request and normalize text and reasoning output."""
        model = kwargs.get("model", "phi4:latest")
        start = time.time()
        self._refresh_concurrency_limit()

        tools = kwargs.get("tools")
        messages = kwargs.get("messages")
        # Structured outputs: Ollama usa o campo ``format`` ("json" ou JSON Schema).
        ollama_format = ptools.to_ollama_format(kwargs.get("response_format"))
        # Migração condicional: só usa /api/chat (messages + tools) quando há tools
        # ou histórico multi-turn; caso contrário mantém /api/generate + reasoning.
        use_chat = bool(tools) or bool(messages)

        # --- LÓGICA DE INJEÇÃO DE THINKING (apenas /api/generate) ---
        is_reasoning_model = any(k in model.lower() for k in REASONING_MODEL_KEYWORDS)
        final_prompt = prompt
        if is_reasoning_model and not use_chat and "<think>" not in prompt:
            final_prompt = (
                "You are a reasoning model. "
                "Please output your thought process within <think> tags before your final answer.\n\n"
                f"{prompt}"
            )

        await self._acquire_slot(model)
        try:
            options = {
                "temperature": kwargs.get("temperature", 0.5),
                "num_predict": kwargs.get("max_tokens", 512),
                "num_ctx": 4096,
            }

            # Quick Win #6: Adaptive timeout based on model type
            explicit_timeout = kwargs.get("timeout_seconds")
            timeout = (
                max(1.0, float(explicit_timeout))
                if explicit_timeout is not None
                else _pa._get_adaptive_timeout(model, workload_class=kwargs.get("workload_class"))
            )
            client = await get_http_client()

            call = self._chat_call if use_chat else self._generate_call
            data, text_out, tool_calls, finish_reason, reasoning = await call(
                client,
                model=model,
                prompt=prompt if use_chat else final_prompt,
                image_b64=image_b64,
                options=options,
                timeout=timeout,
                ollama_format=ollama_format,
                is_reasoning=is_reasoning_model,
                kwargs=kwargs,
            )

            load_ns = data.get("load_duration", 0)
            load_sec = float(load_ns) / 1_000_000_000.0

            p_tok = data.get("prompt_eval_count", 0)
            c_tok = data.get("eval_count", 0)
            cost = get_model_cost(model, p_tok, c_tok)
            latency = time.time() - start
            # Ocupação do equipamento: total_duration do Ollama (ns), senão o tempo de relógio.
            occupancy_s = float(data.get("total_duration", 0) or 0) / 1_000_000_000.0 or latency
            cost_imputed = impute_local_cost(occupancy_s)

            self._record_metrics(model, latency, cost, True)
            self._record_generation_metrics(model, c_tok, latency)
            self._record_load(model, load_sec)

            return LLMResponse(
                text=text_out,
                latency=latency,
                load_time=load_sec,
                cost=cost,
                cost_imputed=cost_imputed,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                model_used=model,
                raw_payload=json.dumps(data),
                reasoning=reasoning,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        except Exception as e:
            self._record_metrics(model, time.time() - start, 0, False)

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
                traceback=traceback.format_exc(),
            )
            raise
        finally:
            self._release_slot(model)
