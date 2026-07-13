# Objective: Shared HTTP query execution for /query and OpenAI-compat routes.
"""Extract query route orchestration from main.py for reuse across endpoints."""

from __future__ import annotations

import asyncio
import json
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
        yield "event: meta\ndata: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
        if tool_calls:
            # Streaming "fake" (pós-resposta) não transmite tool_calls por token;
            # entrega os tool_calls num evento dedicado.
            yield "event: tool_calls\ndata: " + json.dumps({"tool_calls": tool_calls}, ensure_ascii=False) + "\n\n"
        else:
            for token in answer.split():
                yield "event: token\ndata: " + json.dumps({"text": token + " "}, ensure_ascii=False) + "\n\n"
                await asyncio.sleep(0)
        yield "event: done\ndata: " + json.dumps(
            {"status": "completed", "finish_reason": finish_reason}, ensure_ascii=False
        ) + "\n\n"

    return StreamingResponse(_event_gen(), media_type="text/event-stream")
