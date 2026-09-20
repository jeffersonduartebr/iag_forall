# -*- coding: utf-8 -*-
# Objective: anthropic provider adapter (split from providers/_implementations.py).
"""Concrete provider adapter; shared base and shims live in ``providers._base``."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import app.providers_async as _pa
from app import provider_tools as ptools  # type: ignore[attr-defined]
from app.utils.breaker_async import guarded_by

from ._base import (
    BaseProvider,
    LLMResponse,
    get_model_cost,
)
from ._infra import (
    CLOUD_BREAKERS,
    COMMON_RETRY_STRATEGY,
)
from ._timeouts import resolve_timeout


class AnthropicProvider(BaseProvider):
    """Call Anthropic models and normalize the response for router consumers."""

    def __init__(self):
        """Create the Anthropic client and configure cloud-provider concurrency."""
        if _pa.AsyncAnthropic is None:
            raise ImportError("Anthropic SDK not installed")
        # max_retries=0: o retry é do tenacity, por fora. Ver _openai.py.
        self.client = _pa.AsyncAnthropic(api_key=_pa.ANTHROPIC_API_KEY, max_retries=0)
        super().__init__("anthropic", concurrency_limit=50)

    @guarded_by(CLOUD_BREAKERS["anthropic"])
    @COMMON_RETRY_STRATEGY
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        """Execute one Anthropic request and normalize the provider payload."""
        model = kwargs.get("model", "claude-3-5-sonnet-latest")
        start = time.time()

        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")

        await self._acquire_slot(model)
        try:
            canonical_messages = ptools.build_provider_messages(
                prompt, kwargs.get("system_prompt"), kwargs.get("messages"), image_b64
            )
            system_text, anth_messages = ptools.to_anthropic_messages(canonical_messages)

            # Anthropic não tem ``response_format`` nativo: emulamos JSON mode em melhor
            # esforço anexando uma instrução ao system prompt (item #5 do roadmap).
            json_suffix = ptools.json_mode_system_suffix(kwargs.get("response_format"))
            if json_suffix:
                system_text = f"{system_text or ''}{json_suffix}"

            create_args: Dict[str, Any] = {
                "model": model,
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.5),
                "messages": anth_messages,
            }
            if system_text:
                create_args["system"] = system_text
            if tools and not ptools.tools_disabled(tool_choice):
                anth_tools = ptools.to_anthropic_tools(tools)
                if anth_tools:
                    create_args["tools"] = anth_tools
                    anth_tool_choice = ptools.to_anthropic_tool_choice(tool_choice)
                    if anth_tool_choice:
                        create_args["tool_choice"] = anth_tool_choice

            create_args["timeout"] = resolve_timeout(kwargs)
            resp = await self.client.messages.create(**create_args)

            # Itera blocos (text + tool_use) em vez de assumir content[0].text.
            text_out, tool_calls, finish_reason = ptools.from_anthropic_response(resp)
            usage = resp.usage
            p_tok = usage.input_tokens
            c_tok = usage.output_tokens

            cost = get_model_cost(model, p_tok, c_tok)
            latency = time.time() - start
            self._record_metrics(model, latency, cost, True)
            self._record_generation_metrics(model, c_tok, latency)

            return LLMResponse(
                text=text_out,
                latency=latency,
                load_time=0.0,
                cost=cost,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                model_used=model,
                raw_payload=str(resp),
                reasoning=None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        except Exception:
            self._record_metrics(model, time.time() - start, 0, False)
            raise
        finally:
            self._release_slot(model)
