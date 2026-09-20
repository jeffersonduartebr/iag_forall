# -*- coding: utf-8 -*-
# Objective: gemini provider adapter (split from providers/_implementations.py).
"""Concrete provider adapter; shared base and shims live in ``providers._base``."""

from __future__ import annotations

import base64
import json
import time
from types import SimpleNamespace
from typing import Any, Dict, NamedTuple, Optional

import app.providers_async as _pa
from app import provider_tools as ptools  # type: ignore[attr-defined]
from app.utils.breaker_async import guarded_by
from app.utils.executors import run_blocking_provider

from ._base import (
    BaseProvider,
    LLMResponse,
    count_tokens,
    get_model_cost,
)
from ._infra import (
    CLOUD_BREAKERS,
    COMMON_RETRY_STRATEGY,
    google_genai,
)


class _GenOptions(NamedTuple):
    temperature: float
    max_tokens: int
    tools: Optional[list]
    tool_config: Optional[dict]
    system_instruction: Optional[str]
    response_format: Optional[dict]


def _genai_contents(prompt: str, image_b64: Optional[str], contents: Optional[list]) -> list:
    if contents:
        return contents
    parts: list = [{"text": prompt}]
    if image_b64:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
    return parts


def _legacy_contents(prompt: str, image_b64: Optional[str], contents: Optional[list]) -> list:
    if contents:
        return contents
    parts: list = [prompt]
    if image_b64:
        parts.append({"mime_type": "image/jpeg", "data": base64.b64decode(image_b64)})
    return parts


class GeminiProvider(BaseProvider):
    """Call Gemini models through the available Google SDK path."""

    class GeminiAdapter:
        """Isolate SDK-specific Gemini calls behind a small compatibility layer."""

        def generate(
            self,
            model_name: str,
            prompt: str,
            image_b64: Optional[str],
            temperature: float,
            max_tokens: int,
            tools: Optional[list] = None,
            tool_config: Optional[dict] = None,
            contents: Optional[list] = None,
            system_instruction: Optional[str] = None,
            response_format: Optional[dict] = None,
        ):
            """Execute one Gemini generation call using the available SDK.

            Quando ``tools``/``contents`` são informados, retorna o objeto de
            resposta bruto (para ``from_gemini_response`` extrair function calls);
            caso contrário, preserva o comportamento antigo (``SimpleNamespace(text)``).
            """
            options = _GenOptions(temperature, max_tokens, tools, tool_config, system_instruction, response_format)
            if google_genai is not None:
                resp = self._generate_genai(model_name, _genai_contents(prompt, image_b64, contents), options)
                if tools or contents:
                    return resp
                return SimpleNamespace(text=getattr(resp, "text", "") or "")
            if _pa.genai is None:
                raise ImportError("No Gemini SDK available")
            return self._generate_legacy(model_name, _legacy_contents(prompt, image_b64, contents), options)

        @staticmethod
        def _generate_genai(model_name: str, contents: list, options: "_GenOptions"):
            """``google-genai`` SDK: one client call with everything in ``config``."""
            config: Dict[str, Any] = {"temperature": options.temperature, "max_output_tokens": options.max_tokens}
            if options.system_instruction:
                config["system_instruction"] = options.system_instruction
            if options.tools:
                config["tools"] = options.tools
            if options.tool_config:
                config["tool_config"] = options.tool_config
            config.update(ptools.to_gemini_response_config(options.response_format))
            client = google_genai.Client(api_key=_pa.GEMINI_API_KEY or None)
            return client.models.generate_content(model=model_name, contents=contents, config=config)

        @staticmethod
        def _generate_legacy(model_name: str, contents: list, options: "_GenOptions"):
            """Legacy ``google.generativeai`` SDK: tools/system on the model, the rest per call."""
            model_kwargs: Dict[str, Any] = {}
            if options.system_instruction:
                model_kwargs["system_instruction"] = options.system_instruction
            if options.tools:
                model_kwargs["tools"] = options.tools
            generation_config = {"temperature": options.temperature, "max_output_tokens": options.max_tokens}
            generation_config.update(ptools.to_gemini_response_config(options.response_format))
            gen_kwargs: Dict[str, Any] = {"generation_config": generation_config}
            if options.tool_config:
                gen_kwargs["tool_config"] = options.tool_config
            return _pa.genai.GenerativeModel(model_name, **model_kwargs).generate_content(contents, **gen_kwargs)

    def __init__(self):
        """Initialize the Gemini provider and its adapter."""
        if not _pa.genai:
            raise ImportError("Google GenAI SDK not installed")
        super().__init__("gemini", concurrency_limit=60)
        self._adapter = self.GeminiAdapter()

    @guarded_by(CLOUD_BREAKERS["gemini"])
    @COMMON_RETRY_STRATEGY
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        """Execute one Gemini request and normalize the response payload."""
        model_name = kwargs.get("model", "gemini-1.5-flash")
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        messages = kwargs.get("messages")
        start = time.time()

        gem_tools = None
        gem_tool_config = None
        gem_contents = None
        gem_system = None
        if tools and not ptools.tools_disabled(tool_choice):
            gem_tools = ptools.to_gemini_tools(tools)
            gem_tool_config = ptools.to_gemini_tool_config(tool_choice)
        if messages:
            canonical = ptools.build_provider_messages(prompt, kwargs.get("system_prompt"), messages, image_b64)
            gem_system, gem_contents = ptools.to_gemini_contents(canonical)

        await self._acquire_slot(model_name)
        try:

            def _call():
                """Invoke the adapter in a worker thread-friendly callable."""
                return self._adapter.generate(
                    model_name=model_name,
                    prompt=prompt,
                    image_b64=image_b64,
                    temperature=kwargs.get("temperature", 0.5),
                    max_tokens=kwargs.get("max_tokens", 512),
                    tools=gem_tools,
                    tool_config=gem_tool_config,
                    contents=gem_contents,
                    system_instruction=gem_system,
                    response_format=kwargs.get("response_format"),
                )

            # Pool dedicado, não o executor por omissão: o SDK do Gemini é
            # síncrono e a thread não é cancelável, por isso um Gemini lento
            # drenava as 16 threads partilhadas e parava a aplicação inteira.
            resp = await run_blocking_provider(_call)
            text_out, tool_calls, finish_reason = ptools.from_gemini_response(resp)

            p_tok = count_tokens(prompt, model_name)
            # Gemini conta tokens no cliente; num turno de tool o texto é vazio, então
            # contabiliza também os tool_calls serializados para não subestimar custo.
            completion_text = text_out
            if tool_calls:
                completion_text = f"{text_out}{json.dumps(tool_calls, ensure_ascii=False)}"
            c_tok = count_tokens(completion_text, model_name)
            cost = get_model_cost(model_name, p_tok, c_tok)

            latency = time.time() - start
            self._record_metrics(model_name, latency, cost, True)
            self._record_generation_metrics(model_name, c_tok, latency)

            return LLMResponse(
                text=text_out,
                latency=latency,
                load_time=0.0,
                cost=cost,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                model_used=model_name,
                raw_payload=str(resp),
                reasoning=None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        except Exception:
            self._record_metrics(model_name, time.time() - start, 0, False)
            raise
        finally:
            self._release_slot(model_name)
