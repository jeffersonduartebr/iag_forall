# -*- coding: utf-8 -*-
# Objective: Pre-provider stages of the synchronous routing path.
"""Stages that run before the provider call in ``route_and_answer_internal_impl``.

Each stage takes a :class:`RouteContext` (request parameters, injected
dependencies and per-stage timings) and does one thing: semantic-cache
lookup, epistemic uncertainty, candidate resolution, tool-capability
filtering, model selection (incl. OpenRouter exploration and adversarial
escalation) and prompt preparation (RAG). Dependencies keep the same keys as
before (``deps[...]``), so callers and tests inject them unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.model_registry import filter_configured_model_names, filter_tool_capable_model_names
from app.native_tools import filter_native_tool_capable_model_names, split_tools
from app.services.adversarial_governance import advgov_escalate

_CLOUD_PREFIXES = ("openrouter/", "openai/", "anthropic/", "gemini/")


def quietly(action: Callable[[], Any]) -> None:
    """Run a best-effort side effect (metrics); failures never affect routing."""
    try:
        action()
    except Exception:
        pass


@dataclass
class RouteContext:
    """One routing request: parameters, injected dependencies and stage timings."""

    deps: Dict[str, Any]
    query: str
    system_prompt: str
    use_rag: bool
    max_tokens: Any
    temperature: Any
    modality: str
    image_b64: Optional[str]
    rag_modality: str
    use_cache: bool
    runtime_hints: Optional[Dict[str, Any]] = None
    tenant_id: Optional[str] = None
    tools: Optional[list] = None
    tool_choice: Any = None
    messages: Optional[list] = None
    response_format: Optional[dict] = None
    start_time: float = field(default_factory=time.time)
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def hints(self) -> Dict[str, Any]:
        return self.runtime_hints or {}

    @property
    def tools_requested(self) -> bool:
        return bool(self.tools)

    def observe_stage(self, stage: str, started_at: float) -> None:
        """Record stage latency (Prometheus + per-request breakdown for diagnostics)."""
        elapsed = time.time() - started_at
        self.stage_timings_ms[stage] = round(self.stage_timings_ms.get(stage, 0.0) + elapsed * 1000.0, 3)
        quietly(lambda: self.deps["ROUTER_STAGE_LATENCY"].labels(stage=stage).observe(elapsed))


@dataclass
class RouteChoice:
    """Selected model, the top-2 used for hedging, and exploration details."""

    chosen: str
    top2: List[str]
    exploration_mode: bool = False
    exploration_info: Dict[str, Any] = field(default_factory=dict)


def _record_breakers(deps: Dict[str, Any]) -> None:
    deps["_record_dependency_breaker_metrics"]()


async def _lookup_cache(ctx: RouteContext) -> Optional[Dict[str, Any]]:
    deps = ctx.deps
    breaker = deps["_dep_cache_breaker"]
    kwargs = {"modality": ctx.modality, "image_b64": ctx.image_b64, "tenant_id": ctx.tenant_id}
    if breaker.current_state == "open":
        return None
    try:
        return await breaker.call_async(deps["check_cache"], ctx.query, **kwargs)
    except Exception:
        return await deps["check_cache"](ctx.query, **kwargs)


def _cache_hit_result(ctx: RouteContext, cached: Dict[str, Any]) -> Dict[str, Any]:
    deps = ctx.deps
    deps["logger"].info(f"[router] Cache HIT ({cached.get('similarity', 0):.2f})")
    quietly(lambda: deps["ROUTER_ROUTE_COST"].labels(route_type="cache").inc(0.0))
    quietly(lambda: deps["ROUTER_ATTEMPTS_PER_QUERY"].observe(1))
    return {
        "model": "semantic_cache",
        "modality": ctx.modality,
        "answer": cached.get("text", ""),
        "image_output_b64": cached.get("image_output_b64"),
        "latency_s": round(time.time() - ctx.start_time, 3),
        "estimated_cost_usd": 0.0,
        "cost_per_1k": 0.0,
        "metadata": {"cached": True, "stage_timings_ms": dict(ctx.stage_timings_ms)},
        "route": {
            "chosen_model": "semantic_cache",
            "objectives": {"latency": 0, "cost": 0, "uncertainty": 0},
            "pareto_front": [],
            "explanation": "Cache",
            "fallback": {"used": False, "models_tried": ["semantic_cache"], "errors": []},
        },
        "candidates": [],
    }


async def try_cache_hit(ctx: RouteContext) -> Optional[Dict[str, Any]]:
    """Semantic-cache lookup behind its circuit breaker; returns the router result on a hit."""
    if not ctx.use_cache:
        return None
    started = time.time()
    cached = None
    try:
        cached = await _lookup_cache(ctx)
        _record_breakers(ctx.deps)
    except Exception:
        quietly(lambda: ctx.deps["DEPENDENCY_FAILURES"].labels(dependency="cache").inc())
        _record_breakers(ctx.deps)
    finally:
        ctx.observe_stage("cache_lookup", started)
    return _cache_hit_result(ctx, cached) if cached else None


async def compute_uncertainty(ctx: RouteContext) -> float:
    """Epistemic uncertainty u(q) behind its breaker, off the event loop (0.5 on failure)."""
    deps = ctx.deps
    started = time.time()
    score = 0.5
    try:
        breaker = deps["_dep_uq_breaker"]
        if breaker.current_state != "open":
            # Embedding + Redis: roda em thread para não bloquear o event loop.
            score = await deps["asyncio"].to_thread(breaker.call, deps["get_uncertainty_score"], ctx.query, ctx.modality)
        _record_breakers(deps)
    except Exception as exc:
        quietly(lambda: deps["DEPENDENCY_FAILURES"].labels(dependency="uq").inc())
        _record_breakers(deps)
        deps["logger"].warning(f"[router] UQ fail: {exc}")
    finally:
        ctx.observe_stage("precheck", started)
    return score


def _not_blocked(deps: Dict[str, Any], models: Any) -> List[str]:
    """Distinct string model ids (config order) outside the blocked prefixes."""
    prefixes = deps["BLOCKED_PREFIXES"]
    return list(
        dict.fromkeys(m for m in (models or []) if isinstance(m, str) and not any(m.startswith(p) for p in prefixes))
    )


def _configured_candidates(ctx: RouteContext) -> List[str]:
    deps = ctx.deps
    settings = deps["settings"]
    all_candidates = (
        settings.CANDIDATE_MODELS_LIST + settings.CANDIDATE_VISION_MODELS_LIST + settings.CANDIDATE_MULTIMODAL_MODELS_LIST
    )
    models = filter_configured_model_names(_not_blocked(deps, all_candidates))
    if "apply_ollama_performance_preferences" in deps:
        try:
            models = deps["apply_ollama_performance_preferences"](models, runtime_hints=ctx.runtime_hints)
        except Exception as exc:
            deps["logger"].warning(f"[router] Failed to apply Ollama performance preferences: {exc}")
    if ctx.hints.get("prefer_cloud_models"):
        cloud = [m for m in models if not str(m).startswith("ollama/")]
        if cloud:
            models = cloud + [m for m in models if str(m).startswith("ollama/")]
    return models


async def _error_budget_exceeded(deps: Dict[str, Any]) -> bool:
    check_async = deps.get("_is_error_budget_exceeded_async")
    if check_async is not None:
        return await check_async()
    return deps["_is_error_budget_exceeded"]()


async def resolve_candidates(ctx: RouteContext) -> List[str]:
    """Configured candidates, with cloud/local fallbacks and the error-budget local-only mode."""
    deps = ctx.deps
    models = _configured_candidates(ctx)
    if not models:
        cloud_fallback = [
            m for m in (deps["settings"].CANDIDATE_MODELS_LIST or []) if isinstance(m, str) and m.startswith(_CLOUD_PREFIXES)
        ]
        models = filter_configured_model_names(cloud_fallback)
    if not models:
        return ["ollama/phi4:latest"]
    if await _error_budget_exceeded(deps):
        local = [m for m in models if m.startswith("ollama/")]
        if local:
            deps["logger"].warning("[router] Error budget exceeded; forcing local-only candidate set")
            return local
    return models


def restrict_to_tool_models(ctx: RouteContext, models: List[str]) -> List[str]:
    """Strict capability routing: with tools requested, keep only function-calling models."""
    if not ctx.tools_requested:
        return models
    deps = ctx.deps
    configured = _not_blocked(deps, getattr(deps["settings"], "CANDIDATE_TOOL_MODELS_LIST", None))
    if configured:
        candidates = filter_configured_model_names(configured)
    else:
        candidates = filter_tool_capable_model_names(models)
    native_tools = split_tools(ctx.tools)[1]
    if native_tools and candidates:
        candidates = filter_native_tool_capable_model_names(candidates, native_tools)
    if not candidates:
        raise deps["ProviderCallError"](
            model="none",
            message=(
                "Nenhum modelo com suporte a tools está configurado. "
                "Defina CANDIDATE_TOOL_MODELS_LIST ou inclua um modelo com function calling."
            ),
            category="no_tool_model",
            retryable=False,
        )
    return candidates


async def _top2_and_pick(ctx: RouteContext, models: List[str], uncertainty: float) -> Tuple[List[str], str]:
    """NSGA-weighted top-2 followed by the bandit's choice between them."""
    deps = ctx.deps
    weights = await deps["get_dynamic_strategy_weights_async"](ctx.modality)
    top2 = await deps["asyncio"].to_thread(
        deps["choose_top2_models"], models, weights, ctx.query, ctx.modality, uncertainty
    )
    select_async = deps.get("select_model_async")
    if select_async is not None:
        return top2, await select_async(top2, ctx.query, ctx.modality)
    return top2, await deps["asyncio"].to_thread(deps["select_model"], top2, ctx.query, ctx.modality)


async def _try_exploration(ctx: RouteContext, models: List[str], uncertainty: float) -> Optional[RouteChoice]:
    """OpenRouter exploration pick (never with tools), recording the incumbent for shadow comparison."""
    deps = ctx.deps
    pick = deps.get("maybe_pick_openrouter_exploration") if not ctx.tools_requested else None
    if pick is None:
        return None
    try:
        result = await pick(
            known_models=set(models), modality=ctx.modality, settings=deps["settings"], uncertainty_score=uncertainty
        )
    except Exception as exc:
        deps["logger"].warning("[router] OpenRouter exploration pick failed: %s", exc)
        return None
    if not result:
        return None
    chosen, info = result
    try:
        _, info["incumbent_model"] = await _top2_and_pick(ctx, models, uncertainty)
    except Exception as exc:
        deps["logger"].debug("[router] incumbent pick for shadow failed: %s", exc)
    deps["logger"].info("[router] OpenRouter exploration: %s", chosen)
    return RouteChoice(chosen=chosen, top2=[chosen], exploration_mode=True, exploration_info=info)


async def select_route(ctx: RouteContext, models: List[str], uncertainty: float) -> RouteChoice:
    """Exploration when enabled, else top-2 + bandit, then adversarial escalation."""
    explored = await _try_exploration(ctx, models, uncertainty)
    if explored is not None:
        return explored
    top2, chosen = await _top2_and_pick(ctx, models, uncertainty)
    # Governança adversarial (roadmap #17): cluster de alto risco ou incerteza alta
    # pode escalar para um candidato mais forte; no-op sem ADVGOV_ENABLED.
    chosen, top2 = advgov_escalate(ctx.deps, chosen, top2, models, uncertainty, ctx.runtime_hints)
    return RouteChoice(chosen=chosen, top2=top2)


def _empty_bundle(ctx: RouteContext, reason: str, default_mode: str) -> Dict[str, Any]:
    return {
        "citations": [],
        "evidence_snippets": [],
        "grounded": False,
        "knowledge_version": None,
        "retrieval_mode": ctx.hints.get("retrieval_mode", default_mode),
        "retrieval_skipped_reason": reason,
    }


async def _retrieve(ctx: RouteContext) -> Tuple[Optional[str], Dict[str, Any]]:
    """Run retrieval; returns ``(augmented_text, bundle)``."""
    deps = ctx.deps
    rag_mode = ctx.rag_modality if not (ctx.image_b64 and ctx.modality != "text") else ctx.modality
    kwargs = {
        "modality": rag_mode,
        "image_b64": ctx.image_b64,
        "k": int(ctx.hints.get("rag_top_k", 3)),
        "retrieval_mode": ctx.hints.get("retrieval_mode"),
        "context_token_budget": ctx.hints.get("rag_context_token_budget"),
        "rerank_enabled": ctx.hints.get("rag_rerank_enabled"),
    }
    if "build_retrieval_bundle" in deps:
        bundle = await deps["build_retrieval_bundle"](ctx.query, **kwargs)
        return bundle.get("augmented_prompt") or ctx.query, bundle
    bundle = _empty_bundle(ctx, "bypassed", "no_retrieval")
    return await deps["build_augmented_prompt"](ctx.query, **kwargs), bundle


async def prepare_prompt(ctx: RouteContext, skip_rag: bool) -> Tuple[str, Dict[str, Any]]:
    """Final prompt plus retrieval provenance (RAG skipped, successful or degraded)."""
    build = ctx.deps["build_final_prompt"]
    if skip_rag:
        bundle = _empty_bundle(ctx, "runtime_bypass", "no_retrieval")
        return build(query=ctx.query, system_prompt=ctx.system_prompt, use_rag=False, rag_text=None), bundle
    started = time.time()
    try:
        augmented, bundle = await _retrieve(ctx)
        has_context = bool(bundle.get("context"))
        if not has_context:
            bundle["grounded"] = False
        prompt = build(
            query=ctx.query,
            system_prompt=ctx.system_prompt,
            use_rag=has_context,
            rag_text=augmented if has_context else None,
        )
        return prompt, bundle
    except Exception as exc:
        ctx.deps["logger"].warning(f"[router] RAG fail: {exc}")
        prompt = build(query=ctx.query, system_prompt=ctx.system_prompt, use_rag=True, rag_text=None)
        return prompt, _empty_bundle(ctx, "rag_failure", "full_retrieval")
    finally:
        ctx.observe_stage("retrieval", started)
