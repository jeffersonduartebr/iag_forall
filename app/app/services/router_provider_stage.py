# -*- coding: utf-8 -*-
# Objective: Provider-call stage and result assembly of the synchronous routing path.
"""Provider execution (hedged, fallback chain or direct) and router result assembly.

Split out of ``route_and_answer_internal_impl``. The deadline helpers bound
provider timeouts to the remaining synchronous budget; ``execute_provider``
maps hedge/fallback outcomes into a :class:`ProviderOutcome`; ``build_result``
produces the router response contract (unchanged).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.orcamento_tempo import ordem_de_fallback, prazo_da_chamada, reserva_de_fallback
from app.services.router_stages import RouteChoice, RouteContext, quietly


@dataclass
class ProviderOutcome:
    """Provider answer plus which model produced it and what was tried."""

    out: Any
    meta: Any
    chosen: str
    fallback_used: bool = False
    models_tried: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0


def deadline_remaining_seconds(runtime_hints: Optional[Dict[str, Any]]) -> float:
    """Return the remaining synchronous request budget."""
    deadline_ts = float((runtime_hints or {}).get("request_deadline_ts", 0.0) or 0.0)
    if deadline_ts <= 0:
        return float("inf")
    return max(0.0, deadline_ts - time.monotonic())


def effective_provider_timeout_seconds(runtime_hints: Optional[Dict[str, Any]]) -> Optional[float]:
    """Clamp one provider timeout to the remaining synchronous deadline."""
    configured_timeout = float((runtime_hints or {}).get("provider_timeout_seconds", 0.0) or 0.0)
    remaining = deadline_remaining_seconds(runtime_hints)
    if remaining == float("inf"):
        return configured_timeout or None
    budget = max(1.0, remaining - 0.5)
    if configured_timeout <= 0:
        return budget
    return max(1.0, min(configured_timeout, budget))


def hedge_delay_seconds(
    deps: Dict[str, Any], model: str, modality: str, runtime_hints: Optional[Dict[str, Any]]
) -> float:
    """Seconds to wait before launching the hedge (backup) request.

    Fires the backup once the primary exceeds a multiple of its expected (EMA)
    latency — the tied-request heuristic. A fixed ``REQUEST_HEDGE_DELAY_MS`` wins
    when set; otherwise the delay is ``EMA * REQUEST_HEDGE_EMA_FACTOR``. Never
    exceeds the remaining synchronous deadline.
    """
    fixed_ms = deps["_safe_setting_int"]("REQUEST_HEDGE_DELAY_MS", 0)
    if fixed_ms and fixed_ms > 0:
        base = fixed_ms / 1000.0
    else:
        ema = None
        getter = deps.get("get_ema_latency")
        if getter is not None:
            try:
                ema = getter(model, modality)
            except Exception:
                ema = None
        factor = deps["_safe_setting_float"]("REQUEST_HEDGE_EMA_FACTOR", 1.3)
        base = (ema if ema and ema > 0 else 8.0) * factor
    remaining = deadline_remaining_seconds(runtime_hints)
    if remaining != float("inf"):
        base = min(base, max(0.5, remaining - 1.0))
    return max(0.2, base)


def _use_hedging(ctx: RouteContext, choice: RouteChoice) -> bool:
    """Hedging races the primary against one distinct backup (perf #22); never for tool/multi-turn turns."""
    deps, top2 = ctx.deps, choice.top2
    return bool(
        deps["_safe_setting_bool"]("REQUEST_HEDGING_ENABLED", False)
        and "execute_with_hedge" in deps
        and len(top2) >= 2
        and top2[1]
        and top2[1] != choice.chosen
        and not ctx.tools_requested
        and not ctx.messages
    )


def _call_timeout(ctx: RouteContext, model: str) -> Optional[float]:
    """Per-model timeout: the model's measured throughput for the requested tokens, within the deadline."""
    remaining = deadline_remaining_seconds(ctx.runtime_hints)
    configured = effective_provider_timeout_seconds(ctx.runtime_hints)
    if remaining == float("inf"):
        return configured
    return prazo_da_chamada(model, int(ctx.max_tokens), remaining, piso=configured or 0.0)


def _provider_call(ctx: RouteContext, final_prompt: str, retry_empty: bool):
    async def _execute(model_name: str):
        try:
            out, meta = await _call(model_name)
            if retry_empty:
                _raise_if_empty(ctx, model_name, out, meta)
        except Exception:
            _record_provider(ctx, model_name, False)
            raise
        _record_provider(ctx, model_name, True)
        return out, meta

    async def _call(model_name: str):
        return await ctx.deps["call_model"](
            model=model_name,
            prompt=final_prompt,
            modality=ctx.modality,
            image_b64=ctx.image_b64,
            temperature=ctx.temperature,
            max_tokens=ctx.max_tokens,
            timeout_seconds=_call_timeout(ctx, model_name),
            workload_class=ctx.hints.get("workload_class"),
            tools=ctx.tools,
            tool_choice=ctx.tool_choice,
            messages=ctx.messages,
            response_format=ctx.response_format,
            system_prompt=ctx.system_prompt,
        )

    return _execute


def _record_provider(ctx: RouteContext, model: str, ok: bool) -> None:
    record = ctx.deps.get("record_provider_outcome")
    if record is not None:
        quietly(lambda: record(model, ok))


def _raise_if_empty(ctx: RouteContext, model: str, out: Any, meta: Any) -> None:
    """An empty answer (e.g. reasoning ate the token budget) is a failed attempt, so the chain moves on."""
    tool_calls = meta.get("tool_calls") if isinstance(meta, dict) else None
    if tool_calls or (out if isinstance(out, str) else str(out or "")).strip():
        return
    raise ctx.deps["ProviderCallError"](
        model=model, message="Empty answer from provider", category="empty_answer", retryable=True
    )


def _deadline_guard(ctx: RouteContext, model: str) -> None:
    if deadline_remaining_seconds(ctx.runtime_hints) <= 1.0:
        raise ctx.deps["ProviderCallError"](
            model=model,
            message="Synchronous deadline exhausted before provider execution",
            category="provider_timeout",
            retryable=True,
        )


def _count_retries(deps: Dict[str, Any], model: str, errors: List[Dict[str, Any]]) -> None:
    for err in errors:
        deps["ROUTER_RETRY_TOTAL"].labels(model=model, reason=err.get("category", "provider_error")).inc()


def _finalize(ctx: RouteContext, primary: str, result: Any) -> ProviderOutcome:
    """Map a FallbackResult/HedgeResult onto a ProviderOutcome, raising on failure."""
    deps = ctx.deps
    if not result.success:
        quietly(lambda: _count_retries(deps, result.model_used or primary, result.errors))
        last = result.errors[-1] if result.errors else {}
        raise deps["ProviderCallError"](
            model=result.model_used if result.errors else primary,
            message=last.get("error", "All provider attempts failed"),
            category=last.get("category", "provider_unavailable"),
            retryable=True,
        )
    out, meta = result.result
    tried = result.models_tried
    used = len(tried) > 1
    quietly(lambda: _count_retries(deps, result.model_used, result.errors))
    if used and len(tried) >= 2:
        quietly(lambda: deps["FALLBACK_USED"].labels(first_model=tried[0], second_model=tried[1]).inc())
    return ProviderOutcome(out, meta, result.model_used, used, tried, result.errors, max(0, len(tried) - 1))


async def _run_hedged(ctx: RouteContext, choice: RouteChoice, execute) -> ProviderOutcome:
    deps, primary = ctx.deps, choice.chosen
    _deadline_guard(ctx, primary)
    started = time.time()
    hedge = await deps["execute_with_hedge"](
        models=[primary, choice.top2[1]],
        execute_fn=execute,
        hedge_delay_s=hedge_delay_seconds(deps, primary, ctx.modality, ctx.runtime_hints),
        max_parallel=deps["_safe_setting_int"]("REQUEST_HEDGE_MAX_PARALLEL", 2),
    )
    ctx.observe_stage("provider_call", started)
    outcome = _finalize(ctx, primary, hedge)
    label = "backup_win" if outcome.chosen != primary else "primary_win"
    quietly(lambda: deps["ROUTER_HEDGE"].labels(outcome=label).inc())
    return outcome


def _fallback_budget(ctx: RouteContext) -> int:
    """Max fallbacks, or 0 when the remaining deadline cannot fit another attempt."""
    deps = ctx.deps
    max_fallbacks = int(ctx.hints.get("max_fallbacks", deps["_safe_setting_int"]("REQUEST_MAX_FALLBACKS", 2)))
    if deadline_remaining_seconds(ctx.runtime_hints) < reserva_de_fallback():
        quietly(lambda: deps["ROUTER_FALLBACK_SKIPPED"].labels(reason="insufficient_deadline_budget").inc())
        return 0
    return max_fallbacks


async def _run_fallback_chain(ctx: RouteContext, choice: RouteChoice, execute) -> ProviderOutcome:
    _deadline_guard(ctx, choice.chosen)
    max_fallbacks = _fallback_budget(ctx)
    started = time.time()
    result = await ctx.deps["execute_with_fallback"](
        primary_model=choice.chosen,
        execute_fn=execute,
        max_fallbacks=max_fallbacks,
        candidates=ordem_de_fallback(ctx.candidates, choice.chosen),
    )
    ctx.observe_stage("provider_call", started)
    return _finalize(ctx, choice.chosen, result)


async def _run_direct(ctx: RouteContext, choice: RouteChoice, execute) -> ProviderOutcome:
    started = time.time()
    _deadline_guard(ctx, choice.chosen)
    out, meta = await execute(choice.chosen)
    ctx.observe_stage("provider_call", started)
    return ProviderOutcome(out, meta, choice.chosen, models_tried=[choice.chosen])


async def execute_provider(ctx: RouteContext, choice: RouteChoice, final_prompt: str) -> ProviderOutcome:
    """Call the provider: hedged race, fallback chain or a single direct call."""
    # Uma resposta vazia só vira falha quando há outro modelo para tentar; numa chamada directa ela segue
    # para a abstenção (query_reliability), que é o que o cliente sabe tratar.
    hedged = _use_hedging(ctx, choice)
    chained = not hedged and ctx.deps["_safe_setting_bool"]("REQUEST_FALLBACK_ENABLED", False)
    execute = _provider_call(ctx, final_prompt, retry_empty=hedged or chained)
    if hedged:
        return await _run_hedged(ctx, choice, execute)
    if chained:
        return await _run_fallback_chain(ctx, choice, execute)
    return await _run_direct(ctx, choice, execute)


def _parse_cost(ctx: RouteContext, outcome: ProviderOutcome):
    deps = ctx.deps
    started = time.time()
    try:
        return deps["parse_meta_cost"](
            meta=outcome.meta, chosen_model=outcome.chosen, cost_lookup=deps["get_model_cost"]
        )
    except Exception as exc:
        deps["logger"].warning(f"[router] Metadata error: {exc}")
        return 0, 0, 0.0, 0.0, {}
    finally:
        ctx.observe_stage("postprocess", started)


def _record_empty_answer(ctx: RouteContext, outcome: ProviderOutcome, answer: str, tool_calls, meta_safe) -> None:
    # Um turno de tool call legitimamente tem answer vazio — não conta como resposta vazia.
    if answer.strip() or tool_calls:
        return
    meta = outcome.meta if isinstance(outcome.meta, dict) else {}
    reason = "thinking_only" if (meta_safe.get("reasoning") or meta.get("reasoning")) else "empty_content"
    quietly(lambda: ctx.deps["ROUTER_RESPONSE_EMPTY"].labels(model=outcome.chosen, reason=reason).inc())


def build_result(
    ctx: RouteContext,
    choice: RouteChoice,
    outcome: ProviderOutcome,
    uncertainty: float,
    retrieval_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the router response (answer, costs, metadata, route explanation)."""
    deps = ctx.deps
    latency_s = round(time.time() - ctx.start_time, 3)
    p_tok, c_tok, total_cost, load_time_s, meta_safe = _parse_cost(ctx, outcome)
    route_type = "fallback" if outcome.fallback_used else "direct"
    quietly(lambda: deps["ROUTER_ROUTE_COST"].labels(route_type=route_type).inc(float(total_cost)))
    quietly(lambda: deps["ROUTER_ATTEMPTS_PER_QUERY"].observe(max(1, outcome.retry_count + 1)))

    meta = outcome.meta if isinstance(outcome.meta, dict) else {}
    answer = outcome.out if isinstance(outcome.out, str) else str(outcome.out)
    tool_calls = meta.get("tool_calls")
    _record_empty_answer(ctx, outcome, answer, tool_calls, meta_safe)
    chosen = outcome.chosen
    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "finish_reason": meta.get("finish_reason"),
        "model": chosen,
        "modality": ctx.modality,
        "image_output_b64": meta_safe.get("image_output_b64"),
        "latency_s": latency_s,
        "load_time_s": load_time_s,
        "estimated_cost_usd": total_cost,
        "cost_per_1k": total_cost,
        # Custo de caixa (sem a ocupação local imputada): base de orçamento/cobrança.
        "cash_cost_usd": float(meta_safe.get("cash_cost_usd", total_cost) or 0.0),
        "metadata": {
            "raw_payload": meta_safe.get("raw_payload"),
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "cash_cost_usd": meta_safe.get("cash_cost_usd", total_cost),
            "imputed_cost_usd": meta_safe.get("imputed_cost_usd", 0.0),
            "load_time": load_time_s,
            "uncertainty_score": uncertainty,
            "citations": retrieval_bundle.get("citations", []),
            "evidence_snippets": retrieval_bundle.get("evidence_snippets", []),
            "grounded": bool(retrieval_bundle.get("grounded")),
            "knowledge_version": retrieval_bundle.get("knowledge_version"),
            "retrieval_mode": retrieval_bundle.get("retrieval_mode"),
            "retrieval_skipped_reason": retrieval_bundle.get("retrieval_skipped_reason"),
            "openrouter_exploration": choice.exploration_mode,
            "exploration_info": choice.exploration_info,
            "stage_timings_ms": dict(ctx.stage_timings_ms),
        },
        "route": {
            "chosen_model": chosen,
            "modality_selected": ctx.modality,
            "is_multimodal_route": bool(ctx.image_b64),
            "objectives": {"latency": latency_s, "cost": total_cost, "uncertainty": uncertainty},
            "pareto_front": choice.decision.get("pareto_front", []),
            "strategy_weights": choice.decision.get("weights", {}),
            "explanation": (
                f"OpenRouter exploration: {chosen}"
                if choice.exploration_mode
                else f"Selected {chosen} (UQ={uncertainty:.2f})"
            ),
            "fallback": {"used": outcome.fallback_used, "models_tried": outcome.models_tried, "errors": outcome.errors},
        },
        # Os candidatos considerados, com os três objectivos de cada um e a
        # marca de quem está na frente de Pareto. Isto era `[]` literal: sem
        # ele não havia forma de reconstruir *porquê* um modelo foi escolhido,
        # só qual.
        "candidates": choice.decision.get("candidates", []),
        "decision": choice.decision,
    }
