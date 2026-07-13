# Objective: Pure translation between the canonical (OpenAI) tool-calling
# format and each provider's native request/response shape.
"""Camada de tradução de tool/function calling (funções puras, sem I/O).

Este módulo converte, em ambas as direções, entre o **formato canônico** (schema
de tools da OpenAI, usado por OpenAI e OpenRouter nativamente e adotado como
padrão interno) e o formato nativo de cada provedor (Anthropic, Gemini, Ollama).

Não importa SDKs nem faz I/O — por isso é 100% testável isoladamente.

Formatos canônicos
------------------
- ``tools``: ``[{"type":"function","function":{"name","description","parameters":<JSON schema>}}]``
- ``tool_choice``: ``"auto" | "none" | "required" | {"type":"function","function":{"name":...}}``
- ``tool_calls`` (saída): ``[{"id","type":"function","function":{"name","arguments":"<json str>"}}]``
- ``messages``: lista no formato de chat da OpenAI
  ``{"role":"system"|"user"|"assistant"|"tool", "content": str|list|None,
     "tool_calls":[...], "tool_call_id": str, "name": str}``
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.native_tools import is_native_tool, split_tools

ToolCall = Dict[str, Any]
Messages = List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers básicos
# ---------------------------------------------------------------------------
def _new_tool_call_id(prefix: str = "call") -> str:
    """Gera um id sintético de tool_call para provedores que não devolvem um."""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _loads_args(raw: Any) -> Dict[str, Any]:
    """Converte ``arguments`` (str JSON ou dict) em dict, de forma tolerante."""
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            return {}
    # Objetos proto/Map-like (ex.: Gemini) → tenta materializar
    try:
        return dict(raw)
    except Exception:
        return {}


def _dumps_args(raw: Any) -> str:
    """Normaliza ``arguments`` para string JSON (contrato canônico de saída)."""
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(_loads_args(raw), ensure_ascii=False)
    except Exception:
        return "{}"


def canonical_tool_call(name: Optional[str], arguments: Any, call_id: Optional[str] = None) -> ToolCall:
    """Monta um ``tool_call`` no formato canônico (arguments como string JSON)."""
    return {
        "id": call_id or _new_tool_call_id(),
        "type": "function",
        "function": {"name": name or "", "arguments": _dumps_args(arguments)},
    }


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Acessa ``key`` tanto em dicts quanto em objetos de SDK (via getattr)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _function_of(tool: Any) -> Dict[str, Any]:
    """Extrai o bloco ``function`` de um tool canônico (ou o próprio, se plano)."""
    fn = _get(tool, "function")
    if isinstance(fn, dict):
        return fn
    if fn is not None:
        return {"name": _get(fn, "name"), "description": _get(fn, "description"), "parameters": _get(fn, "parameters")}
    return tool if isinstance(tool, dict) else {}


def tools_disabled(tool_choice: Any) -> bool:
    """Indica se o cliente pediu explicitamente para não chamar tools."""
    return isinstance(tool_choice, str) and tool_choice.strip().lower() == "none"


def _data_uri_to_parts(url: str) -> Tuple[str, str]:
    """Separa um data URI base64 em ``(media_type, base64)``."""
    try:
        header, b64 = url.split(",", 1)
        media_type = "image/jpeg"
        if header.startswith("data:") and ";" in header:
            media_type = header[len("data:"):].split(";", 1)[0] or "image/jpeg"
        return media_type, b64
    except Exception:
        return "image/jpeg", url


# ---------------------------------------------------------------------------
# Montagem de mensagens canônicas (fonte da verdade multi-turn)
# ---------------------------------------------------------------------------
def build_provider_messages(
    prompt: str,
    system_prompt: Optional[str] = None,
    messages: Optional[Messages] = None,
    image_b64: Optional[str] = None,
) -> Messages:
    """Retorna a lista de mensagens canônica (formato OpenAI).

    - Se ``messages`` vier preenchido (follow-up multi-turn), ele é a **fonte da
      verdade**; apenas prepende ``system_prompt`` se nenhum system estiver presente.
    - Caso contrário, sintetiza um turno único de ``user`` a partir de ``prompt``
      (embutindo a imagem em ``content`` quando houver), preservando o comportamento
      atual (o ``system`` já vem embutido no ``prompt`` montado pela pipeline).
    """
    if messages:
        out: Messages = [dict(m) for m in messages]
        if system_prompt and not any(_get(m, "role") == "system" for m in out):
            out.insert(0, {"role": "system", "content": system_prompt})
        return out

    out = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})
    if image_b64:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
        out.append({"role": "user", "content": content})
    else:
        out.append({"role": "user", "content": prompt})
    return out


def _content_to_text(content: Any) -> str:
    """Extrai apenas o texto de um ``content`` (str ou lista de partes OpenAI)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content)


def _content_images(content: Any) -> List[str]:
    """Retorna os base64 das imagens presentes em ``content`` (partes image_url)."""
    imgs: List[str] = []
    if isinstance(content, list):
        for p in content:
            if isinstance(p, dict) and p.get("type") == "image_url":
                url = (p.get("image_url") or {}).get("url", "")
                if url:
                    _mt, b64 = _data_uri_to_parts(url)
                    imgs.append(b64)
    return imgs


# ---------------------------------------------------------------------------
# OpenAI / OpenRouter (formato canônico == nativo)
# ---------------------------------------------------------------------------
def serialize_openai_tool_calls(raw_tool_calls: Any) -> Optional[List[ToolCall]]:
    """Converte ``message.tool_calls`` do SDK OpenAI para o formato canônico."""
    if not raw_tool_calls:
        return None
    out: List[ToolCall] = []
    for tc in raw_tool_calls:
        fn = _get(tc, "function", {})
        out.append(
            canonical_tool_call(
                _get(fn, "name"),
                _get(fn, "arguments", "{}"),
                call_id=_get(tc, "id"),
            )
        )
    return out or None


def openai_finish_reason(raw_finish: Optional[str], has_tool_calls: bool) -> str:
    """Normaliza o ``finish_reason`` da OpenAI (tool_calls tem prioridade)."""
    if has_tool_calls:
        return "tool_calls"
    return raw_finish or "stop"


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
def to_anthropic_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Traduz tools canônicas para o formato Anthropic (usa ``input_schema``).

    Function tools são traduzidas (``input_schema``). Tools nativas/server-side (ex.:
    ``{"type":"web_search_20250305","name":"web_search"}``) são repassadas inalteradas,
    ao lado das traduzidas, pois o próprio Anthropic as executa.
    """
    if not tools:
        return None
    out = []
    for t in tools:
        if is_native_tool(t):
            out.append(t)
            continue
        fn = _function_of(t)
        out.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description") or "",
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def to_anthropic_tool_choice(tool_choice: Any) -> Optional[Dict[str, Any]]:
    """Traduz ``tool_choice`` canônico para o formato Anthropic."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return {"auto": {"type": "auto"}, "required": {"type": "any"}}.get(tool_choice.lower())
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        if name:
            return {"type": "tool", "name": name}
    return None


def _anthropic_user_content(content: Any) -> List[Dict[str, Any]]:
    """Converte um ``content`` OpenAI em blocos de conteúdo Anthropic."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks: List[Dict[str, Any]] = []
    if isinstance(content, list):
        for p in content:
            if not isinstance(p, dict):
                blocks.append({"type": "text", "text": str(p)})
            elif p.get("type") == "text":
                blocks.append({"type": "text", "text": p.get("text", "")})
            elif p.get("type") == "image_url":
                media_type, b64 = _data_uri_to_parts((p.get("image_url") or {}).get("url", ""))
                blocks.append(
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}
                )
    return blocks or [{"type": "text", "text": _content_to_text(content)}]


def to_anthropic_messages(messages: Messages) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Traduz mensagens canônicas para ``(system, messages)`` Anthropic."""
    system: Optional[str] = None
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = _get(m, "role")
        content = _get(m, "content")
        if role == "system":
            text = _content_to_text(content)
            system = f"{system}\n\n{text}" if system else text
        elif role == "user":
            out.append({"role": "user", "content": _anthropic_user_content(content)})
        elif role == "assistant":
            blocks: List[Dict[str, Any]] = []
            text = _content_to_text(content)
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in _get(m, "tool_calls") or []:
                fn = _get(tc, "function", {})
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": _get(tc, "id") or _new_tool_call_id(),
                        "name": _get(fn, "name"),
                        "input": _loads_args(_get(fn, "arguments")),
                    }
                )
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": _get(m, "tool_call_id"),
                            "content": _content_to_text(content),
                        }
                    ],
                }
            )
    return system, out


def _anthropic_finish(stop_reason: Optional[str], has_tool_calls: bool) -> str:
    """Mapeia ``stop_reason`` Anthropic para o ``finish_reason`` canônico."""
    if has_tool_calls or stop_reason == "tool_use":
        return "tool_calls"
    if stop_reason == "max_tokens":
        return "length"
    return "stop"


def from_anthropic_response(resp: Any) -> Tuple[str, Optional[List[ToolCall]], str]:
    """Extrai ``(texto, tool_calls, finish_reason)`` de uma resposta Anthropic."""
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    for block in _get(resp, "content", []) or []:
        if _get(block, "type") == "tool_use":
            tool_calls.append(
                canonical_tool_call(_get(block, "name"), _get(block, "input", {}), call_id=_get(block, "id"))
            )
        else:
            # Bloco de texto ("type" == "text" ou não especificado); ignora blocos
            # sem texto (ex.: "thinking").
            txt = _get(block, "text")
            if txt:
                text_parts.append(txt)
    finish = _anthropic_finish(_get(resp, "stop_reason"), bool(tool_calls))
    return "".join(text_parts), (tool_calls or None), finish


# ---------------------------------------------------------------------------
# Gemini (Google)
# ---------------------------------------------------------------------------
_GEMINI_SCHEMA_DROP_KEYS = {"additionalProperties", "$schema", "title", "examples", "default"}


def _clean_gemini_schema(schema: Any) -> Any:
    """Remove chaves de JSON Schema não aceitas pela API do Gemini."""
    if isinstance(schema, dict):
        return {
            k: _clean_gemini_schema(v)
            for k, v in schema.items()
            if k not in _GEMINI_SCHEMA_DROP_KEYS
        }
    if isinstance(schema, list):
        return [_clean_gemini_schema(v) for v in schema]
    return schema


def to_gemini_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Traduz tools canônicas para o ``tools`` do Gemini.

    Function tools viram um bloco ``{"function_declarations":[...]}``. Tools nativas de
    grounding/execução (``{"google_search":{}}``, ``{"code_execution":{}}`` ...) são
    mescladas no payload inalteradas, ao lado do bloco de declarações.
    """
    if not tools:
        return None
    function_tools, native_tools = split_tools(tools)
    out: List[Dict[str, Any]] = []
    if function_tools:
        decls = []
        for t in function_tools:
            fn = _function_of(t)
            decls.append(
                {
                    "name": fn.get("name"),
                    "description": fn.get("description") or "",
                    "parameters": _clean_gemini_schema(fn.get("parameters") or {"type": "object", "properties": {}}),
                }
            )
        out.append({"function_declarations": decls})
    out.extend(native_tools)
    return out or None


def to_gemini_tool_config(tool_choice: Any) -> Optional[Dict[str, Any]]:
    """Traduz ``tool_choice`` canônico para o ``tool_config`` do Gemini."""
    if tool_choice is None:
        return None
    mode = "AUTO"
    allowed: Optional[List[str]] = None
    if isinstance(tool_choice, str):
        mode = {"auto": "AUTO", "required": "ANY", "none": "NONE"}.get(tool_choice.lower(), "AUTO")
    elif isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        if name:
            mode, allowed = "ANY", [name]
    cfg: Dict[str, Any] = {"function_calling_config": {"mode": mode}}
    if allowed:
        cfg["function_calling_config"]["allowed_function_names"] = allowed
    return cfg


def _gemini_parts_from_content(content: Any) -> List[Dict[str, Any]]:
    """Converte um ``content`` OpenAI em ``parts`` do Gemini."""
    if isinstance(content, str):
        return [{"text": content}]
    parts: List[Dict[str, Any]] = []
    if isinstance(content, list):
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append({"text": p.get("text", "")})
            elif isinstance(p, dict) and p.get("type") == "image_url":
                media_type, b64 = _data_uri_to_parts((p.get("image_url") or {}).get("url", ""))
                parts.append({"inline_data": {"mime_type": media_type, "data": b64}})
    return parts or [{"text": _content_to_text(content)}]


def to_gemini_contents(messages: Messages) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Traduz mensagens canônicas para ``(system_instruction, contents)`` do Gemini."""
    system: Optional[str] = None
    contents: List[Dict[str, Any]] = []
    for m in messages:
        role = _get(m, "role")
        content = _get(m, "content")
        if role == "system":
            text = _content_to_text(content)
            system = f"{system}\n\n{text}" if system else text
        elif role == "user":
            contents.append({"role": "user", "parts": _gemini_parts_from_content(content)})
        elif role == "assistant":
            parts: List[Dict[str, Any]] = []
            text = _content_to_text(content)
            if text:
                parts.append({"text": text})
            for tc in _get(m, "tool_calls") or []:
                fn = _get(tc, "function", {})
                parts.append({"function_call": {"name": _get(fn, "name"), "args": _loads_args(_get(fn, "arguments"))}})
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
        elif role == "tool":
            raw = _get(m, "content")
            response_obj = raw if isinstance(raw, dict) else {"result": _content_to_text(raw)}
            contents.append(
                {
                    "role": "user",
                    "parts": [{"function_response": {"name": _get(m, "name") or "tool", "response": response_obj}}],
                }
            )
    return system, contents


def from_gemini_response(resp: Any) -> Tuple[str, Optional[List[ToolCall]], str]:
    """Extrai ``(texto, tool_calls, finish_reason)`` de uma resposta Gemini.

    Funciona para ambos os SDKs (``google.genai`` e ``google.generativeai``): itera
    ``candidates[0].content.parts`` buscando ``function_call``/``text``. Cai de volta
    para ``resp.text`` quando não há candidates estruturados.
    """
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    candidates = _get(resp, "candidates")
    if candidates:
        content = _get(candidates[0], "content")
        for part in _get(content, "parts", []) or []:
            fc = _get(part, "function_call")
            txt = _get(part, "text")
            if fc is not None:
                tool_calls.append(canonical_tool_call(_get(fc, "name"), _get(fc, "args", {})))
            elif txt:
                text_parts.append(txt)
    if not text_parts and not tool_calls:
        txt = _get(resp, "text")
        if txt:
            text_parts.append(txt)
    finish = "tool_calls" if tool_calls else "stop"
    return "".join(text_parts), (tool_calls or None), finish


# ---------------------------------------------------------------------------
# Ollama (endpoint /api/chat)
# ---------------------------------------------------------------------------
def to_ollama_messages(messages: Messages) -> List[Dict[str, Any]]:
    """Traduz mensagens canônicas para o formato de ``/api/chat`` do Ollama."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = _get(m, "role")
        content = _get(m, "content")
        msg: Dict[str, Any] = {"role": role, "content": _content_to_text(content)}
        images = _content_images(content)
        if images:
            msg["images"] = images
        tcs = _get(m, "tool_calls")
        if tcs:
            msg["tool_calls"] = [
                {
                    "function": {
                        "name": _get(_get(tc, "function", {}), "name"),
                        "arguments": _loads_args(_get(_get(tc, "function", {}), "arguments")),
                    }
                }
                for tc in tcs
            ]
        name = _get(m, "name")
        if role == "tool" and name:
            msg["tool_name"] = name
        out.append(msg)
    return out


def from_ollama_chat(data: Dict[str, Any]) -> Tuple[str, Optional[List[ToolCall]], str]:
    """Extrai ``(texto, tool_calls, finish_reason)`` de uma resposta ``/api/chat``."""
    message = (data or {}).get("message", {}) or {}
    text_out = message.get("content", "") or ""
    tool_calls: List[ToolCall] = []
    for tc in message.get("tool_calls") or []:
        fn = (tc or {}).get("function", {}) or {}
        tool_calls.append(canonical_tool_call(fn.get("name"), fn.get("arguments") or {}, call_id=tc.get("id")))
    done_reason = (data or {}).get("done_reason") or ""
    if tool_calls:
        finish = "tool_calls"
    elif done_reason == "length":
        finish = "length"
    else:
        finish = "stop"
    return text_out, (tool_calls or None), finish


# ---------------------------------------------------------------------------
# Structured outputs / JSON mode (response_format) — item #5 do roadmap
# ---------------------------------------------------------------------------
# ``response_format`` canônico (estilo OpenAI):
#   {"type": "json_object"}                                → JSON livre
#   {"type": "json_schema", "json_schema": {"name",         → JSON com schema
#                                            "schema", "strict"}}
#   {"type": "text"} | ausente                              → sem restrição (default)
def _response_format_type(response_format: Any) -> str:
    """Retorna o ``type`` normalizado de um ``response_format`` canônico."""
    if not isinstance(response_format, dict):
        return "text"
    return str(response_format.get("type") or "text").strip().lower()


def _response_format_schema(response_format: Any) -> Optional[Dict[str, Any]]:
    """Extrai o JSON Schema de um ``response_format`` do tipo ``json_schema`` (ou ``None``)."""
    if _response_format_type(response_format) != "json_schema":
        return None
    js = response_format.get("json_schema") if isinstance(response_format, dict) else None
    schema = js.get("schema") if isinstance(js, dict) else None
    return schema if isinstance(schema, dict) else None


def wants_json(response_format: Any) -> bool:
    """Indica se o cliente pediu saída em JSON (``json_object`` ou ``json_schema``)."""
    return _response_format_type(response_format) in {"json_object", "json_schema"}


def to_ollama_format(response_format: Any) -> Optional[Any]:
    """Traduz ``response_format`` para o campo ``format`` do Ollama.

    ``json_object`` → ``"json"``; ``json_schema`` → o dict do JSON Schema;
    ``text``/ausente → ``None`` (sem restrição de formato).
    """
    ftype = _response_format_type(response_format)
    if ftype == "json_object":
        return "json"
    if ftype == "json_schema":
        schema = _response_format_schema(response_format)
        return schema if schema is not None else "json"
    return None


def to_gemini_response_config(response_format: Any) -> Dict[str, Any]:
    """Traduz ``response_format`` para chaves de ``generation_config``/``config`` do Gemini.

    Retorna ``{"response_mime_type": "application/json"}`` (mais ``"response_schema"``
    limpo de chaves não aceitas quando há JSON Schema), ou ``{}`` sem pedido de JSON.
    """
    if not wants_json(response_format):
        return {}
    cfg: Dict[str, Any] = {"response_mime_type": "application/json"}
    schema = _response_format_schema(response_format)
    if schema is not None:
        cfg["response_schema"] = _clean_gemini_schema(schema)
    return cfg


def json_mode_system_suffix(response_format: Any) -> Optional[str]:
    """Instrução de sistema para *emular* JSON mode em provedores sem suporte nativo.

    Anthropic não tem ``response_format`` nativo; este sufixo é anexado ao system
    prompt como melhor esforço. Retorna ``None`` quando não há pedido de JSON.
    """
    if not wants_json(response_format):
        return None
    suffix = "\n\nResponda SOMENTE com JSON válido, sem texto fora do objeto JSON."
    schema = _response_format_schema(response_format)
    if schema is not None:
        try:
            suffix += " O JSON deve seguir este schema: " + json.dumps(schema, ensure_ascii=False)
        except Exception:
            suffix += f" (schema: {schema})"
    return suffix
