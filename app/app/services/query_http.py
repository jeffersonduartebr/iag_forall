# Objective: Shared HTTP query execution for /query and OpenAI-compat routes.
"""Extract query route orchestration from main.py for reuse across endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse

from ..api.auth import AuthContext
from ..correlation import get_correlation_id
from ..observability import (
    API_LATENCY,
    API_REQUESTS,
    CANDIDATE_COST,
    CANDIDATE_LAT,
    COST_BY_PROVIDER,
    ROUTER_CHOSEN,
    TOKENS_INPUT_TOTAL,
    TOKENS_OUTPUT_TOTAL,
    TOTAL_COST_USD,
)
from ..schemas import QueryRequest
from ..services.tenant_context import bind_tenant_to_request
from ..utils.redis_distributed import (
    compute_idempotency_key,
    redis_idempotency_get,
    redis_idempotency_set,
)

logger = logging.getLogger(__name__)


def _main():
    """Lazy import so tests can monkeypatch symbols on app.main."""
    from .. import main

    return main


def _should_proactively_defer_query(req: QueryRequest, request: Request | None) -> tuple[bool, str, str, str]:
    """Delegate to main's defer logic without circular imports at module load."""
    from ..main import _should_proactively_defer_query as _defer

    return _defer(req, request)


async def _resolve_idempotency(
    req: QueryRequest,
    request: Request | None,
) -> Optional[Dict[str, Any]]:
    """Return a cached idempotent response when Redis has one."""
    if request is None:
        return None
    header_key = getattr(request.state, "idempotency_key", None)
    if not header_key:
        return None
    key = compute_idempotency_key(
        tenant_id=req.tenant_id,
        query=req.query,
        modality=req.modality,
        model="",
    )
    composite = f"{header_key}:{key}"
    cached = await redis_idempotency_get(composite)
    if cached and cached.get("status") == "completed":
        return cached.get("body")
    return None


async def _store_idempotency(
    req: QueryRequest,
    request: Request | None,
    body: Dict[str, Any],
) -> None:
    if request is None:
        return
    header_key = getattr(request.state, "idempotency_key", None)
    if not header_key:
        return
    key = compute_idempotency_key(
        tenant_id=req.tenant_id,
        query=req.query,
        modality=req.modality,
        model=str(body.get("model", "")),
    )
    composite = f"{header_key}:{key}"
    await redis_idempotency_set(composite, {"status": "completed", "body": body}, ttl_s=300)


async def execute_query(
    req: QueryRequest,
    request: Request | None = None,
    auth: AuthContext | None = None,
) -> Any:
    """Run the synchronous query path and return QueryResponse dict or JSONResponse."""
    req = bind_tenant_to_request(req, auth)
    cached = await _resolve_idempotency(req, request)
    if cached is not None:
        return cached

    start = time.time()
    API_REQUESTS.inc()

    should_defer, _, _, _ = _should_proactively_defer_query(req, request)
    if should_defer and request is not None and getattr(request.state, "defer_to_query_job", False):
        main = _main()
        queued = main.enqueue_query_job(
            req=req,
            correlation_id=get_correlation_id(),
            reason=str(getattr(request.state, "query_job_reason", "overloaded")),
            pressure_state=str(getattr(request.state, "query_job_pressure_state", "elevated")),
            route_path=request.url.path,
            identity_key=(c.host if (c := getattr(request, "client", None)) else None),
            auth=auth,
        )
        return JSONResponse(status_code=202, content=jsonable_encoder(queued))

    main = _main()
    processed = await main.process_query_request(req)
    result = processed["result"]
    image_input = processed["image_input"]

    duration = time.time() - start
    API_LATENCY.observe(duration)

    chosen_model = result["model"]
    estimated_cost_usd = result.get("estimated_cost_usd", result.get("cost_per_1k", 0.0))
    ROUTER_CHOSEN.labels(model=chosen_model).inc()
    CANDIDATE_COST.observe(estimated_cost_usd)
    CANDIDATE_LAT.observe(result["latency_s"])

    cost_usd = estimated_cost_usd
    TOTAL_COST_USD.inc(cost_usd)
    provider = chosen_model.split("/")[0] if "/" in chosen_model else "unknown"
    COST_BY_PROVIDER.labels(provider=provider).inc(cost_usd)

    metadata = result.get("metadata", {})
    prompt_tokens = metadata.get("prompt_tokens", 0)
    completion_tokens = metadata.get("completion_tokens", 0)
    TOKENS_INPUT_TOTAL.labels(model=chosen_model).inc(prompt_tokens)
    TOKENS_OUTPUT_TOTAL.labels(model=chosen_model).inc(completion_tokens)

    correlation_id = get_correlation_id()
    metadata = result.get("metadata", {})
    metadata["correlation_id"] = correlation_id

    main.record_query_side_effects(req, result, image_input)
    response = main.build_query_response(result, correlation_id)
    body = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    await _store_idempotency(req, request, body)
    return response


async def execute_query_stream(
    req: QueryRequest,
    request: Request | None = None,
    auth: AuthContext | None = None,
) -> Any:
    """Stream query results as SSE (provider pass-through when available)."""
    req = bind_tenant_to_request(req, auth)

    should_defer, _, _, _ = _should_proactively_defer_query(req, request)
    if should_defer and request is not None and getattr(request.state, "defer_to_query_job", False):
        main = _main()
        queued = main.enqueue_query_job(
            req=req,
            correlation_id=get_correlation_id(),
            reason=str(getattr(request.state, "query_job_reason", "overloaded")),
            pressure_state=str(getattr(request.state, "query_job_pressure_state", "elevated")),
            route_path=request.url.path,
            identity_key=(c.host if (c := getattr(request, "client", None)) else None),
            auth=auth,
        )
        return JSONResponse(status_code=202, content=jsonable_encoder(queued))

    # Fast-path: stream tokens de verdade do provedor (item #1) quando a query é
    # texto puro, sem tools/RAG/imagem, e o modelo escolhido é streamável.
    if _real_stream_eligible(req):
        model = await _select_stream_model_safe(req)
        from ..providers_stream import is_streamable

        if is_streamable(model):
            return StreamingResponse(_real_token_gen(req, str(model)), media_type="text/event-stream")

    # Fallback: pseudo-stream da resposta já computada (tools/RAG/visão/provedor
    # sem streaming). Mantém tool_calls e o pipeline completo.
    main = _main()
    processed = await main.process_query_request(req)
    result = processed["result"]
    answer = result.get("answer", "")
    tool_calls = result.get("tool_calls")
    finish_reason = result.get("finish_reason") or ("tool_calls" if tool_calls else "stop")
    payload = {
        "model": result.get("model"),
        "modality": result.get("modality"),
        "finish_reason": finish_reason,
        "tool_calls": tool_calls,
        "metadata": result.get("metadata", {}),
    }

    async def _event_gen():
        yield _sse("meta", payload)
        if tool_calls:
            # Streaming "fake" (pós-resposta) não transmite tool_calls por token;
            # entrega os tool_calls num evento dedicado.
            yield _sse("tool_calls", {"tool_calls": tool_calls})
        else:
            for token in answer.split():
                yield _sse("token", {"text": token + " "})
                await asyncio.sleep(0)
        yield _sse("done", {"status": "completed", "finish_reason": finish_reason})

    return StreamingResponse(_event_gen(), media_type="text/event-stream")


def _sse(event: str, data: Dict[str, Any]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: " + json.dumps(data, ensure_ascii=False) + "\n\n"


def _real_stream_eligible(req: QueryRequest) -> bool:
    """Return whether a request qualifies for real provider-token streaming.

    Only plain text turns stream token-by-token; tool calls, RAG, vision and
    multi-turn histories fall back to the fully-computed pseudo-stream.
    """
    if getattr(req, "tools", None) or getattr(req, "messages", None):
        return False
    if getattr(req, "image_b64", None) or (getattr(req, "images", None) or []):
        return False
    if (getattr(req, "modality", None) or "text").lower() != "text":
        return False
    if getattr(req, "enable_rag_for_answer", False) or getattr(req, "enable_rag_for_image", False):
        return False
    from ..settings_dynamic import settings

    return str(settings.get("STREAMING_REAL_ENABLED", "1")).strip() == "1"


async def _select_stream_model_safe(req: QueryRequest) -> Optional[str]:
    """Pick a streamable model for the fast-path, swallowing selection errors."""
    try:
        from ..providers_stream import select_stream_model

        return await select_stream_model(req.query, "text")
    except Exception as exc:
        logger.warning("[stream] model selection failed: %s", exc)
        return None


async def _real_token_gen(req: QueryRequest, model: str) -> Any:
    """SSE generator that streams real provider tokens and records side-effects."""
    from ..providers_stream import StreamingUnsupportedError, astream_model
    from ..services.governance_runtime import check_runtime_budget_async
    from ..services.hot_path_runtime import check_input_guardrails_async
    from ..settings_dynamic import settings

    started = time.time()
    try:
        decision = await check_input_guardrails_async(req.query)
        if not getattr(decision, "allowed", True):
            yield _sse("error", {"category": "guardrail_block", "reasons": getattr(decision, "reasons", [])})
            yield _sse("done", {"status": "blocked"})
            return
        budget = await check_runtime_budget_async(req.tenant_id)
        if not getattr(budget, "allowed", True):
            yield _sse("error", {"category": "tenant_budget_exceeded", "reason": getattr(budget, "reason", None)})
            yield _sse("done", {"status": "rejected"})
            return

        yield _sse("meta", {"model": model, "modality": "text", "streamed": True})
        parts: list[str] = []
        p_tok = c_tok = 0
        finish = "stop"
        async for ev in astream_model(
            model,
            req.query,
            system_prompt=req.system_prompt or "",
            temperature=req.temperature or settings.TEMPERATURE_DEFAULT,
            max_tokens=req.max_tokens or settings.MAX_TOKENS_DEFAULT,
            timeout_seconds=req.timeout_seconds,
        ):
            if ev.type == "delta":
                parts.append(ev.text)
                yield _sse("token", {"text": ev.text})
            elif ev.type == "final":
                p_tok, c_tok, finish = ev.prompt_tokens, ev.completion_tokens, ev.finish_reason or "stop"

        latency = round(time.time() - started, 3)
        _record_stream_side_effects(req, model, "".join(parts), latency, p_tok, c_tok)
        yield _sse("done", {"status": "completed", "finish_reason": finish, "latency_s": latency})
    except StreamingUnsupportedError:
        yield _sse("error", {"category": "stream_unsupported"})
        yield _sse("done", {"status": "error"})
    except Exception as exc:
        logger.warning("[stream] real streaming failed for %s: %s", model, exc)
        yield _sse("error", {"category": "stream_error", "message": str(exc)})
        yield _sse("done", {"status": "error"})


def _record_stream_side_effects(
    req: QueryRequest, model: str, answer: str, latency: float, p_tok: int, c_tok: int
) -> None:
    """Record usage and dispatch the async judge/bandit feedback for a streamed turn."""
    try:
        from ..providers_async import get_model_cost

        cost = float(get_model_cost(model, int(p_tok or 0), int(c_tok or 0)) or 0.0)
    except Exception:
        cost = 0.0
    try:
        from ..services.governance_runtime import schedule_runtime_usage

        schedule_runtime_usage(
            tenant_id=req.tenant_id,
            cost_usd=cost,
            tokens_in=int(p_tok or 0),
            tokens_out=int(c_tok or 0),
            requests=1,
        )
    except Exception as exc:
        logger.warning("[stream] usage record failed: %s", exc)
    if not answer.strip():
        return
    try:
        from ..tasks import task_process_feedback

        task_process_feedback.delay(
            query=req.query,
            answer=answer,
            chosen_model=model,
            modality="text",
            latency_s=latency,
            cost_val=cost,
            image_b64=None,
            raw_payload={"streamed": True},
            prompt_tokens=int(p_tok or 0),
            completion_tokens=int(c_tok or 0),
        )
    except Exception as exc:
        logger.warning("[stream] feedback dispatch failed: %s", exc)
