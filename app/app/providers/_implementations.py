# Objective: Concrete provider adapters (OpenAI/OpenRouter/Anthropic/Gemini/Ollama) (roadmap #19).
"""Provider classes extracted from providers_async. Test-patched hot symbols
(get_http_client, get_model_cost, count_tokens, _runtime_provider_settings, genai)
are routed through the facade module so pa.* monkeypatches keep working."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
import traceback
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

import app.providers_async as _pa
from app import provider_tools as ptools  # type: ignore[attr-defined]
from app.observability import logger as structlog_logger

from ._infra import (
    COMMON_RETRY_STRATEGY,
    OLLAMA_HOST,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_HTTP_REFERER,
    REASONING_MODEL_KEYWORDS,
    cloud_breaker,
    google_genai,
    local_breaker,
)
from ._ollama import (
    _mark_ollama_model_state,
    _ollama_concurrency_controller,
)

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
    cost: float
    prompt_tokens: int
    completion_tokens: int
    model_used: str
    raw_payload: Optional[str] = None
    reasoning: Optional[str] = None  # <--- NOVO CAMPO: Armazena o pensamento (CoT)
    tool_calls: Optional[List[Dict[str, Any]]] = None  # Tool/function calls no formato canônico (OpenAI)
    finish_reason: Optional[str] = None  # "stop" | "tool_calls" | "length" | ...


# ==============================================================================
# 5. ARQUITETURA BASE
# ==============================================================================


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
            if self.name == "ollama":
                _mark_ollama_model_state(model, inflight_delta=-1)
            try:
                _pa.PROVIDER_INFLIGHT_REQUESTS.labels(model=model).dec()
            except Exception:
                pass

    def _record_metrics(self, model: str, latency: float, cost: float, success: bool):
        """Publish provider-level success, latency, and cost metrics."""
        _pa.PROV_REQ.labels(model=model).inc()
        if success:
            _pa.PROV_OK.labels(model=model).inc()
            _pa.PROV_LAT.labels(model=model).observe(latency)
            _pa.PROV_COST.labels(model=model).observe(cost)
        else:
            _pa.PROV_ERR.labels(model=model).inc()

    def _record_generation_metrics(self, model: str, completion_tokens: int, latency: float) -> None:
        """Publish generation throughput metrics for successful responses."""
        if latency <= 0 or completion_tokens <= 0:
            return
        try:
            _pa.GENERATION_TOKENS_PER_SECOND.labels(model=model).observe(completion_tokens / latency)
        except Exception:
            pass


# ==============================================================================
# 6. PROVEDOR: OPENAI
# ==============================================================================


class OpenAIProvider(BaseProvider):
    """Call OpenAI chat models through the async SDK and normalize the result."""

    def __init__(self):
        """Create the OpenAI client and configure cloud-provider concurrency."""
        if _pa.AsyncOpenAI is None:
            raise ImportError("OpenAI SDK not installed")
        self.client = _pa.AsyncOpenAI(api_key=_pa.OPENAI_API_KEY)
        super().__init__("openai", concurrency_limit=100)

    @COMMON_RETRY_STRATEGY
    @cloud_breaker
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
                api_args["max_completion_tokens"] = max_tokens
            else:
                api_args["max_tokens"] = max_tokens
                api_args["temperature"] = temperature

            resp = await self.client.chat.completions.create(**api_args)

            choice = resp.choices[0]
            text_out = choice.message.content or ""
            tool_calls = ptools.serialize_openai_tool_calls(getattr(choice.message, "tool_calls", None))
            finish_reason = ptools.openai_finish_reason(getattr(choice, "finish_reason", None), bool(tool_calls))
            usage = resp.usage
            p_tok = usage.prompt_tokens if usage else 0
            c_tok = usage.completion_tokens if usage else 0

            cost = get_model_cost(model, p_tok, c_tok)
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
                reasoning=None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        except Exception as e:
            self._record_metrics(model, time.time() - start, 0, False)
            structlog_logger.error("openai_provider_fail", error=str(e), model=model)
            raise
        finally:
            self._release_slot(model)


# ==============================================================================
# 6b. PROVEDOR: OPENROUTER (OpenAI-compatible gateway)
# ==============================================================================


class OpenRouterProvider(OpenAIProvider):
    """Call models via OpenRouter using the OpenAI-compatible chat API."""

    def __init__(self):
        if _pa.AsyncOpenAI is None:
            raise ImportError("OpenAI SDK not installed")
        self._api_key = ""
        self._refresh_client()
        BaseProvider.__init__(self, "openrouter", concurrency_limit=100)

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
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
        )
        self._api_key = api_key

    @COMMON_RETRY_STRATEGY
    @cloud_breaker
    async def generate(self, prompt: str, image_b64: Optional[str] = None, **kwargs) -> LLMResponse:
        from app.openrouter_catalog import get_openrouter_api_key

        current_key = get_openrouter_api_key()
        if current_key != self._api_key:
            self._refresh_client()
        if not current_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        return await super().generate(prompt, image_b64=image_b64, **kwargs)


# ==============================================================================
# 7. PROVEDOR: ANTHROPIC
# ==============================================================================


class AnthropicProvider(BaseProvider):
    """Call Anthropic models and normalize the response for router consumers."""

    def __init__(self):
        """Create the Anthropic client and configure cloud-provider concurrency."""
        if _pa.AsyncAnthropic is None:
            raise ImportError("Anthropic SDK not installed")
        self.client = _pa.AsyncAnthropic(api_key=_pa.ANTHROPIC_API_KEY)
        super().__init__("anthropic", concurrency_limit=50)

    @COMMON_RETRY_STRATEGY
    @cloud_breaker
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


# ==============================================================================
# 8. PROVEDOR: GEMINI (GOOGLE)
# ==============================================================================


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
            structured = bool(tools or contents)
            if google_genai is not None:
                client = google_genai.Client(api_key=_pa.GEMINI_API_KEY or None)
                if contents:
                    req_contents = contents
                else:
                    req_contents = [{"text": prompt}]
                    if image_b64:
                        req_contents.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
                config: Dict[str, Any] = {"temperature": temperature, "max_output_tokens": max_tokens}
                if system_instruction:
                    config["system_instruction"] = system_instruction
                if tools:
                    config["tools"] = tools
                if tool_config:
                    config["tool_config"] = tool_config
                config.update(ptools.to_gemini_response_config(response_format))
                resp = client.models.generate_content(
                    model=model_name,
                    contents=req_contents,
                    config=config,
                )
                if structured:
                    return resp
                return SimpleNamespace(text=getattr(resp, "text", "") or "")

            if _pa.genai is None:
                raise ImportError("No Gemini SDK available")

            model_kwargs: Dict[str, Any] = {}
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction
            if tools:
                model_kwargs["tools"] = tools
            gmodel = _pa.genai.GenerativeModel(model_name, **model_kwargs)
            if contents:
                gen_contents: Any = contents
            else:
                gen_contents = [prompt]
                if image_b64:
                    gen_contents.append({"mime_type": "image/jpeg", "data": base64.b64decode(image_b64)})
            gen_kwargs: Dict[str, Any] = {
                "generation_config": {"temperature": temperature, "max_output_tokens": max_tokens}
            }
            gen_kwargs["generation_config"].update(ptools.to_gemini_response_config(response_format))
            if tool_config:
                gen_kwargs["tool_config"] = tool_config
            return gmodel.generate_content(gen_contents, **gen_kwargs)

    def __init__(self):
        """Initialize the Gemini provider and its adapter."""
        if not _pa.genai:
            raise ImportError("Google GenAI SDK not installed")
        super().__init__("gemini", concurrency_limit=60)
        self._adapter = self.GeminiAdapter()

    @COMMON_RETRY_STRATEGY
    @cloud_breaker
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

            resp = await asyncio.to_thread(_call)
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


# ==============================================================================
# 9. PROVEDOR: OLLAMA (LOCAL & HTTPX ASYNC)
# ==============================================================================


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
