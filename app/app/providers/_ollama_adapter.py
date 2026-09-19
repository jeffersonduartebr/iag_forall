# -*- coding: utf-8 -*-
# Objective: ollama_adapter provider adapter (split from providers/_implementations.py).
"""Concrete provider adapter; shared base and shims live in ``providers._base``."""

from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from typing import Any, Dict, Optional

import httpx

import app.providers_async as _pa
from app import provider_tools as ptools  # type: ignore[attr-defined]
from app.observability import logger as structlog_logger
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

        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
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

            tool_calls = None
            finish_reason = "stop"

            if use_chat:
                canonical_messages = ptools.build_provider_messages(
                    prompt, kwargs.get("system_prompt"), messages, image_b64
                )
                chat_payload: Dict[str, Any] = {
                    "model": model,
                    "messages": ptools.to_ollama_messages(canonical_messages),
                    "stream": False,
                    "options": options,
                }
                if tools and not ptools.tools_disabled(tool_choice):
                    chat_payload["tools"] = tools
                if ollama_format is not None:
                    chat_payload["format"] = ollama_format
                resp = await client.post(f"{self.host}/api/chat", json=chat_payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                text_out, tool_calls, finish_reason = ptools.from_ollama_chat(data)
                reasoning = None
            else:
                payload = {
                    "model": model,
                    "prompt": final_prompt,
                    "stream": False,
                    # Avoid empty final answers on models that support a separate
                    # thinking channel (for example qwen3.5) unless we explicitly
                    # want reasoning output.
                    "think": is_reasoning_model,
                    "options": options,
                }
                if image_b64:
                    payload["images"] = [image_b64]
                if ollama_format is not None:
                    payload["format"] = ollama_format
                resp = await client.post(f"{self.host}/api/generate", json=payload, timeout=timeout)
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
            # Ocupação do equipamento: total_duration do Ollama (ns), senão o tempo de relógio.
            occupancy_s = float(data.get("total_duration", 0) or 0) / 1_000_000_000.0 or latency
            cost_imputed = impute_local_cost(occupancy_s)

            self._record_metrics(model, latency, cost, True)
            self._record_generation_metrics(model, c_tok, latency)
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
