# Objective: Service-layer helpers for query runtime.
"""Query orchestration helpers extracted from the HTTP layer."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException

from ..ab_testing import get_ab_test_manager
from ..error_handling import ErrorCategory, create_error_response, log_error
from ..guardrails import sanitize_output_guardrails
from ..observability import (
    POLICY_VERSION_ACTIVE,
    QUERY_POLICY_APPLIED,
    ROUTER_QUERY_OUTCOME,
    logger,
)
from ..providers_async import ProviderCallError, ProviderCircuitOpenError
from ..roadmap_features import create_response_review
from ..router_core import route_and_answer
from ..services.feedback_payload import build_feedback_payload
from ..services.governance_degradation import resolve_budget, resolve_policy
from ..services.governance_runtime import (
    check_runtime_budget_async,
    get_runtime_active_policy_async,
    schedule_runtime_usage,
)
from ..services.hot_path_runtime import check_input_guardrails_async
from ..services.tool_governance import audit_tool_denial, evaluate_tool_policy
from ..services.tool_observability import record_tool_turn
from ..settings_dynamic import settings
from ..tasks import task_process_feedback
from .query_profile import (  # noqa: F401  (reexportados para tasks/tests)
    _effective_sync_timeout_seconds,
    _infer_retrieval_profile,
    _workload_provider_timeout_seconds,
    _workload_sync_deadline_seconds,
    apply_query_runtime_profile,
    classify_query_workload,
)
from .query_reliability import _clamp, enrich_result_reliability  # noqa: F401  (reexportados)

check_tenant_budget = check_runtime_budget_async
get_active_policy = get_runtime_active_policy_async
record_tenant_usage = schedule_runtime_usage

def _raise_if_guardrail_blocked(req: Any, decision: Any) -> None:
    """HTTP 400 when the input guardrails block the query."""
    if decision.allowed:
        return
    ROUTER_QUERY_OUTCOME.labels(
        outcome="guardrail_blocked",
        model="guardrails",
        modality=(getattr(req, "modality", None) or "text").lower(),
    ).inc()
    raise HTTPException(
        status_code=400,
        detail={
            "error": True,
            "category": "guardrail_block",
            "message": "Conteúdo bloqueado por política de segurança.",
            "reasons": decision.reasons,
        },
    )


def _raise_if_budget_exceeded(req: Any, budget: Any) -> None:
    """HTTP 429 when the tenant budget is exhausted."""
    if budget.allowed:
        return
    ROUTER_QUERY_OUTCOME.labels(
        outcome="budget_rejected",
        model="budget_control",
        modality=(getattr(req, "modality", None) or "text").lower(),
    ).inc()
    raise HTTPException(
        status_code=429,
        detail={
            "error": True,
            "category": "tenant_budget_exceeded",
            "reason": budget.reason,
            "daily_spent": budget.daily_spent,
            "monthly_spent": budget.monthly_spent,
            "daily_limit": budget.daily_limit,
            "monthly_limit": budget.monthly_limit,
        },
    )


def _resolve_modality(req: Any) -> Tuple[str, Optional[str]]:
    """Requested modality and the image input (first of ``images``); text + image => vision."""
    modality = (req.modality or "text").lower()
    image_input = req.image_b64 or (req.images[0] if req.images else None)
    if image_input and modality == "text":
        modality = "vision"
    return modality, image_input


def _select_policy(req: Any, active_policy: Optional[Dict[str, Any]]) -> Optional[str]:
    """Explicit policy version, else the active one (published as a gauge)."""
    version = (active_policy or {}).get("version")
    if version:
        POLICY_VERSION_ACTIVE.labels(policy_version=str(version)).set(1)
    return req.policy_version or version


def _enforce_tool_policy(req: Any, active_policy: Optional[Dict[str, Any]], modality: str) -> None:
    """HTTP 403 when a requested tool is not allowed by the tenant/tool policy."""
    req_tools = getattr(req, "tools", None)
    if not req_tools:
        return
    decision = evaluate_tool_policy(req_tools, active_policy, req.tenant_id)
    if decision.allowed:
        return
    audit_tool_denial(req.tenant_id, decision)
    ROUTER_QUERY_OUTCOME.labels(outcome="tool_policy_denied", model="tool_governance", modality=modality).inc()
    raise HTTPException(
        status_code=403,
        detail={
            "error": True,
            "category": "tool_policy_denied",
            "reason": decision.reason,
            "tool": decision.offending_tool,
        },
    )


def _assign_experiment(req: Any, selected_policy: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """A/B assignment (variant may override the policy version); failures keep the request going."""
    if not (req.experiment_id and settings.AB_TESTING_ENABLED):
        return None, selected_policy
    try:
        assignment = get_ab_test_manager().get_assignment(
            req.experiment_id,
            req.user_key or req.tenant_id or f"anon:{hash(req.query)}",
        )
    except Exception as exc:
        logger.warning(f"[query] Failed experiment assignment: {exc}")
        return None, selected_policy
    if not assignment:
        return None, selected_policy
    variant_name, variant_cfg = assignment
    return {"name": variant_name, "config": variant_cfg}, variant_cfg.get("policy_version", selected_policy)


_PROVIDER_ERROR_CATEGORIES = {
    "provider_timeout": (ErrorCategory.PROVIDER_TIMEOUT, 504),
    "provider_rate_limit": (ErrorCategory.PROVIDER_RATE_LIMIT, 429),
    "provider_unavailable": (ErrorCategory.PROVIDER_UNAVAILABLE, 502),
}


def _routing_http_error(exc: Exception, modality: str) -> HTTPException:
    """Map a routing failure onto the HTTP error contract (and count the outcome)."""
    if isinstance(exc, asyncio.TimeoutError):
        ROUTER_QUERY_OUTCOME.labels(outcome="provider_timeout", model="unknown", modality=modality).inc()
        info = log_error(asyncio.TimeoutError("Request timed out"), category=ErrorCategory.PROVIDER_TIMEOUT)
        return HTTPException(status_code=504, detail=create_error_response(info))
    if isinstance(exc, ProviderCircuitOpenError):
        ROUTER_QUERY_OUTCOME.labels(outcome="provider_unavailable", model=exc.model or "unknown", modality=modality).inc()
        info = log_error(exc, category=ErrorCategory.CIRCUIT_OPEN, model=exc.model)
        return HTTPException(status_code=503, detail=create_error_response(info))
    if isinstance(exc, ProviderCallError):
        ROUTER_QUERY_OUTCOME.labels(outcome=exc.category, model=exc.model or "unknown", modality=modality).inc()
        if exc.category == "no_tool_model":
            return HTTPException(
                status_code=422, detail={"error": True, "category": "no_tool_model", "message": str(exc)}
            )
        category, status_code = _PROVIDER_ERROR_CATEGORIES.get(
            exc.category, (ErrorCategory.PROVIDER_UNAVAILABLE, 502)
        )
        info = log_error(exc, category=category, model=exc.model)
        return HTTPException(status_code=status_code, detail=create_error_response(info))
    ROUTER_QUERY_OUTCOME.labels(outcome="provider_unavailable", model="unknown", modality=modality).inc()
    info = log_error(exc)
    logger.exception(f"[router] Erro: {exc}")
    return HTTPException(status_code=500, detail=create_error_response(info))


async def _route(req: Any, modality: str, image_input: Optional[str], profile: Dict[str, Any]) -> Dict[str, Any]:
    """Call the router with the runtime profile; failures become HTTP errors."""
    try:
        return await route_and_answer(
            query=req.query,
            system_prompt=req.system_prompt or "",
            use_rag=profile["use_rag"],
            max_tokens=profile["max_tokens"],
            temperature=req.temperature or settings.TEMPERATURE_DEFAULT,
            modality=modality,
            image_b64=image_input,
            rag_modality=(req.rag_modality or "text").lower(),
            use_cache=req.use_cache,
            timeout_seconds=_effective_sync_timeout_seconds(req.timeout_seconds, profile),
            runtime_hints=profile["runtime_hints"],
            tenant_id=req.tenant_id,
            tools=getattr(req, "tools", None),
            tool_choice=getattr(req, "tool_choice", None),
            messages=getattr(req, "messages", None),
            response_format=getattr(req, "response_format", None),
        )
    except Exception as exc:
        raise _routing_http_error(exc, modality) from exc


def _annotate_result(
    req: Any,
    result: Dict[str, Any],
    profile: Dict[str, Any],
    selected_policy: Optional[str],
    assigned_variant: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Output guardrails, governance/experiment metadata and reliability hints."""
    answer_clean, output_tags = sanitize_output_guardrails(result.get("answer", ""))
    result["answer"] = answer_clean
    metadata = result.setdefault("metadata", {})
    metadata.update(
        {
            "guardrail_output_tags": output_tags,
            "policy_version": selected_policy,
            "experiment_id": req.experiment_id,
            "experiment_variant": assigned_variant,
            "tenant_id": req.tenant_id,
            "workload_class": profile["workload_class"],
            "detected_complexity": profile.get("detected_complexity"),
            "perf_mode_enabled": profile["perf_mode_enabled"],
            "retrieval_mode": profile["runtime_hints"]["retrieval_mode"],
        }
    )
    return enrich_result_reliability(result)


def _record_query_cost(result: Dict[str, Any], profile: Dict[str, Any]) -> None:
    try:
        from ..observability import ROUTER_QUERY_COST_USD

        cost_usd = float(result.get("estimated_cost_usd", result.get("cost_per_1k", 0.0)) or 0.0)
        if cost_usd > 0:
            ROUTER_QUERY_COST_USD.labels(
                benchmark_theme=str(profile["runtime_hints"].get("benchmark_theme") or "unknown"),
                detected_complexity=str(profile.get("detected_complexity") or "unknown"),
            ).inc(cost_usd)
    except Exception:
        pass


def _query_outcome(result: Dict[str, Any]) -> str:
    """Outcome label for router_query_outcome_total."""
    if result.get("model", "unknown") == "semantic_cache":
        return "cache_hit"
    if result.get("finish_reason") == "tool_calls" or result.get("tool_calls"):
        return "tool_calls"
    if not str(result.get("answer", "") or "").strip():
        return "empty_answer"
    if bool(result.get("abstained")) and str(result.get("abstain_reason") or "") == "empty_answer":
        return "empty_answer"
    if ((result.get("route") or {}).get("fallback") or {}).get("used"):
        return "fallback_success"
    return "success"


async def process_query_request(req: Any) -> Dict[str, Any]:
    """Process one query request with governance, guardrails, and experimentation hooks."""
    if not req or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query obrigatória.")

    # Guardrails (regex, sem I/O) primeiro: uma requisição bloqueada não toca Redis/DB.
    _raise_if_guardrail_blocked(req, await check_input_guardrails_async(req.query))
    # Orçamento e política ativa são leituras independentes: em paralelo, avaliadas na
    # ordem original (o 429 do orçamento tem precedência sobre uma falha ao ler a política).
    pre_budget, active_policy = await asyncio.gather(
        check_tenant_budget(req.tenant_id), get_active_policy(), return_exceptions=True
    )
    # Um blip do MariaDB não pode derrubar um caminho que não precisa dele.
    _raise_if_budget_exceeded(req, resolve_budget(pre_budget, req.tenant_id))
    active_policy = resolve_policy(active_policy)

    modality, image_input = _resolve_modality(req)
    profile = apply_query_runtime_profile(req, modality=modality, image_input=image_input)
    selected_policy = _select_policy(req, active_policy)
    _enforce_tool_policy(req, active_policy, modality)
    assigned_variant, selected_policy = _assign_experiment(req, selected_policy)
    logger.info(
        "[query] '%s...' (mod=%s, tenant=%s, policy=%s, exp=%s)",
        req.query[:60],
        modality,
        req.tenant_id or "-",
        selected_policy or "-",
        req.experiment_id or "-",
    )

    routed = await _route(req, modality, image_input, profile)
    result = _annotate_result(req, routed, profile, selected_policy, assigned_variant)
    if selected_policy:
        QUERY_POLICY_APPLIED.labels(policy_version=str(selected_policy)).inc()
    _record_query_cost(result, profile)
    ROUTER_QUERY_OUTCOME.labels(
        outcome=_query_outcome(result),
        model=result.get("model", "unknown"),
        modality=result.get("modality", modality),
    ).inc()
    return {
        "result": result,
        "image_input": image_input,
        "selected_policy": selected_policy,
        "assigned_variant": assigned_variant,
        "modality": modality,
    }


def record_query_side_effects(req: Any, result: Dict[str, Any], image_input: str | None) -> None:
    """Persist asynchronous feedback, tenant usage, and experiment metrics."""
    chosen_model = result["model"]
    cost_usd = result.get("estimated_cost_usd", result.get("cost_per_1k", 0))
    metadata = result.get("metadata", {})
    prompt_tokens = metadata.get("prompt_tokens", 0)
    completion_tokens = metadata.get("completion_tokens", 0)
    perf_mode_enabled = str(settings.get("ROUTER_PERF_MODE", "0")).strip() == "1"
    combined_payload = build_feedback_payload(
        result, tenant_id=req.tenant_id, include_raw=not perf_mode_enabled
    )

    # Turno de tool call não tem resposta em texto para julgar: pular juízes/reward
    # (evita envenenar os juízes com um "answer" vazio). Uso/cobrança são mantidos abaixo.
    is_tool_turn = str(result.get("finish_reason")) == "tool_calls" or bool(result.get("tool_calls"))
    if not is_tool_turn:
        try:
            task_process_feedback.delay(
                query=req.query,
                answer=result["answer"],
                chosen_model=chosen_model,
                modality=result["modality"],
                latency_s=result["latency_s"],
                cost_val=result.get("estimated_cost_usd", result.get("cost_per_1k", 0.0)),
                image_b64=image_input,
                raw_payload=combined_payload,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as exc:
            logger.error(f"[main] Falha ao despachar tarefa Celery: {exc}")
    else:
        # Turno de tool: emite métricas e alimenta o bandit com um reward de
        # boa-formação (item #2), já que os juízes foram pulados acima.
        try:
            _msgs = getattr(req, "messages", None)
            record_tool_turn(
                chosen_model=chosen_model,
                tool_calls=result.get("tool_calls") or [],
                modality=result["modality"],
                latency_s=result.get("latency_s", 0.0),
                cost_val=result.get("estimated_cost_usd", result.get("cost_per_1k", 0.0)),
                query=req.query,
                conversation_depth=len(_msgs) if isinstance(_msgs, (list, tuple)) else 0,
                prompt_tokens=int(prompt_tokens or 0),
                completion_tokens=int(completion_tokens or 0),
            )
        except Exception as exc:
            logger.warning(f"[main] Falha ao registrar turno de tool: {exc}")

    try:
        record_tenant_usage(
            tenant_id=req.tenant_id,
            # Orçamento do tenant: só custo de caixa, nunca a ocupação local imputada.
            cost_usd=float(result.get("cash_cost_usd", cost_usd) or 0.0),
            tokens_in=int(prompt_tokens or 0),
            tokens_out=int(completion_tokens or 0),
            requests=1,
        )
    except Exception as exc:
        logger.warning(f"[query] Failed to record tenant usage: {exc}")

    if combined_payload.get("review_status") == "needs_review":
        confidence = combined_payload.get("confidence_score")
        verification = combined_payload.get("verification_status")
        try:
            create_response_review(
                correlation_id=metadata.get("correlation_id"),
                tenant_id=req.tenant_id,
                query_text=req.query,
                answer=result["answer"],
                chosen_model=chosen_model,
                confidence_score=float(confidence) if confidence is not None else None,
                confidence_band=str(combined_payload["confidence_band"] or "") or None,
                grounded=combined_payload["grounded"],
                verification_status=str(verification) if verification else None,
                review_reason=str(combined_payload.get("abstain_reason") or verification or "low_confidence"),
                metadata={
                    "citations": combined_payload["citations"],
                    "evidence_snippets": combined_payload["evidence_snippets"],
                    "knowledge_version": combined_payload.get("knowledge_version"),
                    "workload_class": combined_payload.get("workload_class"),
                },
            )
        except Exception as exc:
            logger.warning(f"[query] Failed to enqueue response review: {exc}")

    if req.experiment_id and settings.AB_TESTING_ENABLED and not perf_mode_enabled:
        try:
            manager = get_ab_test_manager()
            variant = (metadata.get("experiment_variant") or {}).get("name")
            if variant:
                manager.record_result(req.experiment_id, variant, "quality", float(metadata.get("quality", 0.0) or 0.0))
                manager.record_result(req.experiment_id, variant, "latency", float(result.get("latency_s", 0.0)))
                manager.record_result(req.experiment_id, variant, "cost", float(cost_usd))
        except Exception as exc:
            logger.warning(f"[query] Failed to record experiment metrics: {exc}")
