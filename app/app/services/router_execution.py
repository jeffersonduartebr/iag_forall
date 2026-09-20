# Objective: Service-layer helpers for router execution.
"""Fast-path routing implementation extracted from router_core.

``route_and_answer_internal_impl`` orchestrates the stages in
``services.router_stages`` (cache, uncertainty, candidates, selection, prompt)
and ``services.router_provider_stage`` (provider call, result assembly).
"""

from __future__ import annotations

import time
from typing import Any, Dict

from app.services.router_cache_stage import try_cache_hit
from app.services.router_provider_stage import build_result, execute_provider
from app.services.router_services import spawn_via_deps
from app.services.router_stages import (
    RouteChoice,
    RouteContext,
    compute_uncertainty,
    prepare_prompt,
    resolve_candidates,
    restrict_to_tool_models,
    select_route,
)


def _setting_value(settings: Any, key: str, default: Any) -> Any:
    """Return one setting from either the dynamic settings facade or a simple stub object."""
    getter = getattr(settings, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(settings, key, default)


def _should_bypass_rag(
    *,
    deps: Dict[str, Any],
    query: str,
    modality: str,
    image_b64: str | None,
    use_rag: bool,
    uncertainty_score: float,
    runtime_hints: Dict[str, Any] | None,
) -> bool:
    """Return whether retrieval should be skipped for the current request."""
    if not use_rag:
        return True
    if image_b64 or modality in {"vision", "multimodal"}:
        return False
    retrieval_mode = (runtime_hints or {}).get("retrieval_mode")
    if retrieval_mode == "no_retrieval":
        return True
    if (runtime_hints or {}).get("needs_retrieval") is False:
        return True
    if retrieval_mode in {"light_retrieval", "full_retrieval"}:
        return False
    if str(_setting_value(deps["settings"], "RAG_SIMPLE_QUERY_BYPASS_ENABLED", "1")).strip() != "1":
        return False

    query_text = str(query or "").strip()
    token_count = len(query_text.split())
    if uncertainty_score > 0.45:
        return False
    if token_count <= 18 and len(query_text) <= 140:
        return True
    return False


def _ensure_local_model(ctx: RouteContext, chosen: str) -> None:
    """Warm/verify an unverified Ollama model in the background (tracked, bounded)."""
    deps = ctx.deps
    name = chosen.replace("ollama/", "")
    if chosen.startswith("ollama/") and not deps["is_ollama_model_verified"](name):
        spawn_via_deps(
            deps,
            deps["asyncio"].to_thread(deps["_ensure_ollama_model"], name),
            name="ollama_ensure_model",
            limit=2,
        )


async def _choose_route(ctx: RouteContext, uncertainty: float) -> RouteChoice:
    """Candidates -> tool capability filter -> model selection (one 'selection' stage)."""
    started = time.time()
    models = restrict_to_tool_models(ctx, await resolve_candidates(ctx))
    choice = await select_route(ctx, models, uncertainty)
    ctx.observe_stage("selection", started)
    ctx.deps["logger"].info(
        f"[router] Model: {choice.chosen} | UQ: {uncertainty:.2f} | exploration={choice.exploration_mode}"
    )
    return choice


async def route_and_answer_internal_impl(
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
    runtime_hints: Dict[str, Any] | None = None,
    tenant_id: str | None = None,
    tools: list | None = None,
    tool_choice: Any = None,
    messages: list | None = None,
    response_format: dict | None = None,
) -> Dict[str, Any]:
    """Execute the synchronous routing path using injected dependencies."""
    ctx = RouteContext(
        deps=deps,
        query=query,
        system_prompt=system_prompt,
        use_rag=use_rag,
        max_tokens=max_tokens or deps["settings"].MAX_TOKENS_DEFAULT,
        temperature=temperature or deps["settings"].TEMPERATURE_DEFAULT,
        modality=deps["normalize_modality"](modality, image_b64),
        image_b64=image_b64,
        rag_modality=rag_modality,
        # Turnos de tool (tools declaradas ou histórico multi-turn) não são cacheáveis:
        # a resposta depende de resultados de tool específicos do turno.
        use_cache=use_cache and not (tools or messages),
        runtime_hints=runtime_hints,
        tenant_id=tenant_id,
        tools=tools,
        tool_choice=tool_choice,
        messages=messages,
        response_format=response_format,
    )
    cached = await try_cache_hit(ctx)
    if cached is not None:
        return cached

    uncertainty = await compute_uncertainty(ctx)
    choice = await _choose_route(ctx, uncertainty)
    _ensure_local_model(ctx, choice.chosen)

    # Follow-up multi-turn (messages) é a fonte da verdade: não há "query" única
    # para aumentar, então o RAG é ignorado (grounding só se aplica ao 1º turno).
    skip_rag = bool(messages) or _should_bypass_rag(
        deps=deps,
        query=query,
        modality=ctx.modality,
        image_b64=image_b64,
        use_rag=use_rag,
        uncertainty_score=float(uncertainty or 0.0),
        runtime_hints=runtime_hints,
    )
    final_prompt, retrieval_bundle = await prepare_prompt(ctx, skip_rag)
    outcome = await execute_provider(ctx, choice, final_prompt)
    return build_result(ctx, choice, outcome, uncertainty, retrieval_bundle)
