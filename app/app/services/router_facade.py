# Objective: Public facade for route_and_answer with retry, dedup, and timeout handling.
"""High-level routing entrypoint extracted from router_core."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional

from ..observability import ROUTER_SYNC_DEADLINE_EXCEEDED
from ..reliability import get_request_deduplicator
from ..services.router_execution import route_and_answer_internal_impl
from ..services.router_resilience import (
    record_request_outcome as _record_request_outcome_impl,
)
from ..services.router_resilience import (
    safe_setting_int as _safe_setting_int_impl,
)
from ..services.router_services import should_enable_dedup

logger = logging.getLogger(__name__)


def _settings_getter(settings: Any, key: str, default: Any) -> Any:
    getter = getattr(settings, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(settings, key, default)


async def route_and_answer_with_resilience(
    *,
    settings: Any,
    deps: Dict[str, Any],
    internal_coro_factory: Callable[[Dict[str, Any]], Any],
    query: str,
    timeout_seconds: Optional[int],
    runtime_hints: Dict[str, Any] | None,
    deduplicate: bool,
    max_tokens: int | None,
    temperature: float | None,
    system_prompt: str,
) -> Dict[str, Any]:
    """Execute routing with deduplication, retries, timeout, and outcome metrics."""
    default_timeout = _safe_setting_int_impl(lambda k, d: _settings_getter(settings, k, d), "REQUEST_TIMEOUT_SECONDS", 120)
    effective_timeout = timeout_seconds or default_timeout
    effective_runtime_hints = dict(runtime_hints or {})
    effective_runtime_hints["request_deadline_ts"] = time.monotonic() + float(effective_timeout)

    async def _execute_request():
        return await internal_coro_factory(effective_runtime_hints)

    max_retries = _safe_setting_int_impl(lambda k, d: _settings_getter(settings, k, d), "REQUEST_MAX_RETRIES", 1)
    dedup_enabled = should_enable_dedup(lambda k, d=None: _settings_getter(settings, k, d), deduplicate)

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            if dedup_enabled:
                deduplicator = get_request_deduplicator()
                result = await asyncio.wait_for(
                    deduplicator.deduplicate(
                        query=query,
                        model="auto",
                        execute_fn=_execute_request,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system_prompt=system_prompt,
                    ),
                    timeout=effective_timeout,
                )
            else:
                result = await asyncio.wait_for(_execute_request(), timeout=effective_timeout)

            _record_request_outcome_impl(settings_getter=lambda k, d: _settings_getter(settings, k, d), success=True)
            return result
        except asyncio.TimeoutError:
            _record_request_outcome_impl(settings_getter=lambda k, d: _settings_getter(settings, k, d), success=False)
            try:
                ROUTER_SYNC_DEADLINE_EXCEEDED.labels(
                    workload_class=str(effective_runtime_hints.get("workload_class", "unknown"))
                ).inc()
            except Exception:
                pass
            logger.error("[router] Request timeout after %ss for query: %s...", effective_timeout, query[:60])
            raise
        except Exception as exc:
            last_exc = exc
            _record_request_outcome_impl(settings_getter=lambda k, d: _settings_getter(settings, k, d), success=False)
            if attempt >= max_retries:
                raise
            await asyncio.sleep(min(1.5, 0.25 * (2**attempt)))

    if last_exc:
        raise last_exc
    raise RuntimeError("route_and_answer_with_resilience exhausted without result")


def build_internal_route_coro(
    *,
    deps: Dict[str, Any],
    query: str,
    system_prompt: str,
    use_rag: bool,
    max_tokens: int | None,
    temperature: float | None,
    modality: str,
    image_b64: str | None,
    rag_modality: str,
    use_cache: bool,
    tenant_id: str | None,
    tools: list | None = None,
    tool_choice: Any = None,
    messages: list | None = None,
    response_format: dict | None = None,
):
    """Return async callable that invokes route_and_answer_internal_impl."""

    async def _run(runtime_hints: Dict[str, Any]):
        return await route_and_answer_internal_impl(
            deps=deps,
            query=query,
            system_prompt=system_prompt,
            use_rag=use_rag,
            max_tokens=max_tokens,
            temperature=temperature,
            modality=modality,
            image_b64=image_b64,
            rag_modality=rag_modality,
            use_cache=use_cache,
            runtime_hints=runtime_hints,
            tenant_id=tenant_id,
            tools=tools,
            tool_choice=tool_choice,
            messages=messages,
            response_format=response_format,
        )

    return _run
