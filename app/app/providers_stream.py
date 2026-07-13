# -*- coding: utf-8 -*-
# Objective: Real provider-token streaming for the /query/stream fast-path.
"""Streaming real de tokens do provedor (roadmap item #1).

O endpoint ``/query/stream`` antes computava a resposta inteira e a picava por
espaços (pseudo-streaming). Aqui transmitimos tokens de verdade conforme o provedor
os gera, para **OpenAI-compat** (OpenAI, OpenRouter), **Ollama** local e
**Anthropic**. Provedores cujo SDK não expõe streaming assíncrono conveniente
(ex.: Gemini, que roda síncrono via ``to_thread``) levantam
``StreamingUnsupportedError`` para o chamador cair no caminho não-streaming.

Reusa a construção de cliente/credenciais/mensagens de ``providers_async``
(``ProviderFactory``, ``get_http_client``, ``provider_tools``) — sem duplicar
configuração de provedor. A seleção de modelo (``select_stream_model``) reaproveita
os mesmos pesos NSGA + snapshot do bandit da rota síncrona; é uma seleção mais leve
(sem circuit breaker/RAG/UQ) adequada ao caminho rápido de streaming.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from app import provider_tools as ptools  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

# Famílias de provedor que sabemos transmitir token a token.
STREAMABLE_FAMILIES = ("openai", "openrouter", "ollama", "anthropic")


class StreamingUnsupportedError(RuntimeError):
    """Raised when the target model/provider cannot stream; caller should fall back."""


@dataclass
class StreamEvent:
    """One streaming event: incremental text ("delta") or the terminal summary ("final")."""

    type: str
    text: str = ""
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


def provider_family(model: str) -> str:
    """Return the provider namespace prefix (defaults to ``ollama`` when unqualified)."""
    return model.split("/", 1)[0] if "/" in model else "ollama"


def _real_name(model: str) -> str:
    """Strip the provider prefix to the provider-native model id."""
    return model.split("/", 1)[1] if "/" in model else model


def is_streamable(model: Optional[str]) -> bool:
    """Return whether a fully-qualified model can be token-streamed by this module."""
    if not model:
        return False
    return provider_family(model) in STREAMABLE_FAMILIES


async def select_stream_model(query: str, modality: str = "text") -> Optional[str]:
    """Pick a model for the streaming fast-path, reusing NSGA weights + bandit snapshot.

    Lighter than the full router (no circuit breaker / RAG / uncertainty gating); the
    caller falls back to the full pipeline when this returns ``None`` or a
    non-streamable model.
    """
    from app.router_strategy import choose_top2_models
    from app.services.router_strategy_weights import get_dynamic_strategy_weights_async
    from app.settings_dynamic import settings

    candidates = [m for m in (settings.CANDIDATE_MODELS_LIST or []) if isinstance(m, str)]
    if not candidates:
        return None
    weights = await get_dynamic_strategy_weights_async(modality)
    top = choose_top2_models(candidates, weights, query, modality=modality)
    return top[0] if top else None


async def astream_model(
    model: str,
    prompt: str,
    *,
    system_prompt: str = "",
    temperature: float = 0.5,
    max_tokens: int = 512,
    messages: Optional[List[Dict[str, Any]]] = None,
    image_b64: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> AsyncIterator[StreamEvent]:
    """Yield ``StreamEvent`` deltas then one ``final`` event for the given model.

    Dispatches per provider family. Raises ``StreamingUnsupportedError`` for families
    without an async streaming path so the caller can fall back to non-streaming.
    """
    family = provider_family(model)
    if family in ("openai", "openrouter"):
        gen = _stream_openai(model, prompt, system_prompt, temperature, max_tokens, messages, image_b64)
    elif family == "ollama":
        gen = _stream_ollama(model, prompt, system_prompt, temperature, max_tokens, timeout_seconds)
    elif family == "anthropic":
        gen = _stream_anthropic(model, prompt, system_prompt, temperature, max_tokens, messages, image_b64)
    else:
        raise StreamingUnsupportedError(f"streaming not supported for provider '{family}'")
    async for event in gen:
        yield event


async def _stream_openai(
    model: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    messages: Optional[List[Dict[str, Any]]],
    image_b64: Optional[str],
) -> AsyncIterator[StreamEvent]:
    """Stream an OpenAI-compatible chat completion (OpenAI / OpenRouter)."""
    from app.providers_async import ProviderFactory

    client = getattr(ProviderFactory.get_provider(model), "client")
    real = _real_name(model)
    api_messages = ptools.build_provider_messages(prompt, system_prompt or None, messages, image_b64)
    args: Dict[str, Any] = {
        "model": real,
        "messages": api_messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if real.startswith("o1-") or "gpt-5" in real:
        args["max_completion_tokens"] = max_tokens
    else:
        args["max_tokens"] = max_tokens
        args["temperature"] = temperature

    finish = "stop"
    p_tok = c_tok = 0
    stream = await client.chat.completions.create(**args)
    async for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if choices:
            delta = getattr(choices[0], "delta", None)
            piece = getattr(delta, "content", None) if delta is not None else None
            if piece:
                yield StreamEvent("delta", text=piece)
            fr = getattr(choices[0], "finish_reason", None)
            if fr:
                finish = fr
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            p_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            c_tok = int(getattr(usage, "completion_tokens", 0) or 0)
    yield StreamEvent("final", finish_reason=finish, prompt_tokens=p_tok, completion_tokens=c_tok, model=model)


async def _stream_ollama(
    model: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: Optional[float],
) -> AsyncIterator[StreamEvent]:
    """Stream a local Ollama completion via ``/api/generate`` (NDJSON, plain text)."""
    from app.providers_async import OLLAMA_HOST, get_http_client

    real = _real_name(model)
    payload: Dict[str, Any] = {
        "model": real,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 4096},
    }
    if system_prompt:
        payload["system"] = system_prompt
    timeout = max(1.0, float(timeout_seconds)) if timeout_seconds else 120.0

    client = await get_http_client()
    p_tok = c_tok = 0
    async with client.stream("POST", f"{OLLAMA_HOST}/api/generate", json=payload, timeout=timeout) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except (ValueError, TypeError):
                continue
            piece = data.get("response", "")
            if piece:
                yield StreamEvent("delta", text=piece)
            if data.get("done"):
                p_tok = int(data.get("prompt_eval_count", 0) or 0)
                c_tok = int(data.get("eval_count", 0) or 0)
    yield StreamEvent("final", finish_reason="stop", prompt_tokens=p_tok, completion_tokens=c_tok, model=model)


async def _stream_anthropic(
    model: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    messages: Optional[List[Dict[str, Any]]],
    image_b64: Optional[str],
) -> AsyncIterator[StreamEvent]:
    """Stream an Anthropic message via the SDK's ``messages.stream`` helper."""
    from app.providers_async import ProviderFactory

    client = getattr(ProviderFactory.get_provider(model), "client")
    real = _real_name(model)
    canonical = ptools.build_provider_messages(prompt, system_prompt or None, messages, image_b64)
    system_text, anth_messages = ptools.to_anthropic_messages(canonical)
    args: Dict[str, Any] = {
        "model": real,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": anth_messages,
    }
    if system_text:
        args["system"] = system_text

    p_tok = c_tok = 0
    async with client.messages.stream(**args) as stream:
        async for text in stream.text_stream:
            if text:
                yield StreamEvent("delta", text=text)
        final = await stream.get_final_message()
        usage = getattr(final, "usage", None)
        if usage is not None:
            p_tok = int(getattr(usage, "input_tokens", 0) or 0)
            c_tok = int(getattr(usage, "output_tokens", 0) or 0)
    yield StreamEvent("final", finish_reason="stop", prompt_tokens=p_tok, completion_tokens=c_tok, model=model)
