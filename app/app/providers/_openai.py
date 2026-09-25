# -*- coding: utf-8 -*-
# Objective: openai provider adapter (split from providers/_implementations.py).
"""Concrete provider adapter; shared base and shims live in ``providers._base``."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import app.providers_async as _pa
from app import provider_tools as ptools  # type: ignore[attr-defined]
from app.observability import logger as structlog_logger
from app.services.orcamento_tempo import orcamento_raciocinio, tokens_totais
from app.utils.breaker_async import guarded_by

from ._base import (
    BaseProvider,
    LLMResponse,
    get_model_cost,
)
from ._infra import (
    CLOUD_BREAKERS,
    COMMON_RETRY_STRATEGY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_HTTP_REFERER,
)
from ._timeouts import resolve_timeout


class OpenAIProvider(BaseProvider):
    """Call OpenAI chat models through the async SDK and normalize the result."""

    def __init__(self):
        """Create the OpenAI client and configure cloud-provider concurrency."""
        if _pa.AsyncOpenAI is None:
            raise ImportError("OpenAI SDK not installed")
        # max_retries=0: o tenacity já repete por fora (5 tentativas) e o
        # breaker conta uma falha por pedido. Com os 2 retries internos do SDK
        # por dentro disso, uma requisição do utilizador podia gerar até 15
        # chamadas ao upstream — todas facturadas.
        self.client = _pa.AsyncOpenAI(api_key=_pa.OPENAI_API_KEY, max_retries=0)
        super().__init__("openai", concurrency_limit=100)

    # Breaker POR FORA do retry: um pedido do utilizador conta uma falha,
    # não cinco. E `guarded_by` em vez de `@cloud_breaker`, que na versão
    # síncrona do pybreaker nunca chegava a ver a excepção de um async def.
    @guarded_by(CLOUD_BREAKERS["openai"])
    @COMMON_RETRY_STRATEGY
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        """Execute one OpenAI chat-completion request and normalize its output."""
        model = kwargs.get("model", "gpt-4o")
        temperature = kwargs.get("temperature", 0.5)
        max_tokens = kwargs.get("max_tokens", 512)
        start = time.time()

        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")

        await self._acquire_slot(model)
        try:
            api_args = {
                "model": model,
                "messages": ptools.build_provider_messages(
                    prompt, kwargs.get("system_prompt"), kwargs.get("messages"), image_b64
                ),
            }

            # Tools/function calling (formato OpenAI é o canônico → pass-through).
            if tools and not ptools.tools_disabled(tool_choice):
                api_args["tools"] = tools
                if tool_choice is not None:
                    api_args["tool_choice"] = tool_choice

            # Structured outputs / JSON mode (formato OpenAI é o canônico → pass-through
            # nativo, tanto json_object quanto json_schema).
            response_format = kwargs.get("response_format")
            if response_format:
                api_args["response_format"] = response_format

            if model.startswith("o1-") or "gpt-5" in model:
                api_args["max_completion_tokens"] = tokens_totais(max_tokens)
            else:
                api_args["max_tokens"] = max_tokens
                api_args["temperature"] = temperature
            api_args.update(self._reasoning_args(model, max_tokens))
            if "max_tokens" in api_args:
                api_args.pop("max_completion_tokens", None)  # um teto só: o do provedor que reescreveu os args

            # Sem isto o SDK usa o seu default de 600 s e o `timeout_seconds`
            # calculado pelo router era simplesmente ignorado.
            api_args["timeout"] = resolve_timeout(kwargs)
            resp = await self.client.chat.completions.create(**api_args)

            choice = resp.choices[0]
            text_out = choice.message.content or ""
            reasoning = getattr(choice.message, "reasoning", None)  # OpenRouter devolve o raciocínio à parte
            tool_calls = ptools.serialize_openai_tool_calls(getattr(choice.message, "tool_calls", None))
            finish_reason = ptools.openai_finish_reason(getattr(choice, "finish_reason", None), bool(tool_calls))
            usage = resp.usage
            p_tok = usage.prompt_tokens if usage else 0
            c_tok = usage.completion_tokens if usage else 0

            # O OpenRouter devolve o custo real (com desconto de cache e raciocínio cobrado); o catálogo é o fallback.
            cost_real = (getattr(usage, "model_extra", None) or {}).get("cost") if usage else None
            cost = float(cost_real) if isinstance(cost_real, (int, float)) else get_model_cost(model, p_tok, c_tok)
            latency = time.time() - start
            self._record_metrics(model, latency, cost, True)
            self._record_generation_metrics(model, c_tok, latency)

            try:
                raw_payload = json.dumps(resp.model_dump(), default=str)
            except Exception:
                raw_payload = str(resp)

            return LLMResponse(
                text=text_out,
                latency=latency,
                load_time=0.0,
                cost=cost,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                model_used=model,
                raw_payload=raw_payload,
                reasoning=reasoning if isinstance(reasoning, str) and reasoning else None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        except Exception as e:
            self._record_metrics(model, time.time() - start, 0, False)
            structlog_logger.error("openai_provider_fail", error=str(e), model=model)
            raise
        finally:
            self._release_slot(model)

    def _reasoning_args(self, model: str, max_tokens: int) -> Dict[str, Any]:
        """OpenAI reasoning models take the quota inside ``max_completion_tokens`` (set above)."""
        return {}


class OpenRouterProvider(OpenAIProvider):
    """Call models via OpenRouter using the OpenAI-compatible chat API."""

    def __init__(self):
        if _pa.AsyncOpenAI is None:
            raise ImportError("OpenAI SDK not installed")
        self._api_key = ""
        self._refresh_client()
        BaseProvider.__init__(self, "openrouter", concurrency_limit=100)

    def _reasoning_args(self, model: str, max_tokens: int) -> Dict[str, Any]:
        """Reasoning quota, output ceiling and provider policy for one OpenRouter call.

        - OpenRouter counts reasoning inside ``max_tokens``, so the ceiling is answer + quota. OpenAI models only
          accept ``reasoning.effort`` (not ``max_tokens``); the others take the token budget. Models that do
          not reason ignore the param (https://openrouter.ai/docs/use-cases/reasoning-tokens).
        - ``data_collection: deny`` routes only to providers that do not retain or train on prompts: the traffic
          carries minors' submissions (LGPD) (https://openrouter.ai/docs/features/provider-routing).
        """
        extra: Dict[str, Any] = {"provider": {"data_collection": "deny"}}
        args: Dict[str, Any] = {"extra_body": extra}
        budget = orcamento_raciocinio()
        if budget:
            args["max_tokens"] = tokens_totais(max_tokens)
            openai = str(model).startswith("openai/")
            extra["reasoning"] = {"effort": "medium"} if openai else {"max_tokens": budget}
        return args

    def _refresh_client(self) -> None:
        from app.openrouter_catalog import get_openrouter_api_key

        api_key = get_openrouter_api_key()
        default_headers: Dict[str, str] = {}
        referer = (os.getenv("OPENROUTER_HTTP_REFERER", "") or OPENROUTER_HTTP_REFERER or "").strip()
        title = (os.getenv("OPENROUTER_APP_NAME", "") or OPENROUTER_APP_NAME or "").strip()
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-Title"] = title
        base_url = (
            os.getenv("OPENROUTER_BASE_URL", "") or OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1"
        ).strip()
        self.client = _pa.AsyncOpenAI(
            max_retries=0,  # ver acima: o retry é do tenacity, não do SDK
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
        )
        self._api_key = api_key

    # Breaker próprio: o OpenRouter falhar não pode abrir o circuito da OpenAI.
    @guarded_by(CLOUD_BREAKERS["openrouter"])
    @COMMON_RETRY_STRATEGY
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        from app.openrouter_catalog import get_openrouter_api_key

        current_key = get_openrouter_api_key()
        if current_key != self._api_key:
            self._refresh_client()
        if not current_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        return await super().generate(prompt, image_b64=image_b64, **kwargs)
