# Objective: Service-layer helpers for query runtime.
"""Query orchestration helpers extracted from the HTTP layer."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Dict

from fastapi import HTTPException

from ..ab_testing import get_ab_test_manager
from ..error_handling import ErrorCategory, create_error_response, log_error
from ..guardrails import check_input_guardrails, sanitize_output_guardrails
from ..observability import (
    QUERY_POLICY_APPLIED,
    POLICY_VERSION_ACTIVE,
    ROUTER_QUERY_OUTCOME,
    logger,
)
from ..providers_async import ProviderCallError, ProviderCircuitOpenError
from ..router_core import route_and_answer
from ..services.governance_runtime import check_runtime_budget, get_runtime_active_policy, record_runtime_usage
from ..settings_dynamic import settings
from ..tasks import task_process_feedback

check_tenant_budget = check_runtime_budget
get_active_policy = get_runtime_active_policy
record_tenant_usage = record_runtime_usage

_REASONING_HINTS = (
    "step by step",
    "passo a passo",
    "prove",
    "derive",
    "demonstre",
    "justify",
    "explique detalhadamente",
    "chain of thought",
)
_RETRIEVAL_HINTS = (
    "according to",
    "de acordo com",
    "cite",
    "reference",
    "document",
    "manual",
    "policy",
    "regulation",
    "source",
    "baseado no material",
)


def classify_query_workload(req: Any, modality: str, image_input: str | None) -> str:
    """Classify a request into a small set of performance-relevant workload classes."""
    if image_input or modality in {"vision", "multimodal"}:
        return "vision"

    query = str(getattr(req, "query", "") or "").strip()
    lowered = query.lower()

    if any(hint in lowered for hint in _RETRIEVAL_HINTS):
        return "retrieval_heavy"
    if any(hint in lowered for hint in _REASONING_HINTS):
        return "reasoning"

    token_count = len(re.findall(r"\w+", query))
    if token_count <= 18 and len(query) <= 140 and "?" in query:
        return "simple_text"
    if token_count <= 12 and len(query) <= 100:
        return "simple_text"
    return "reasoning"


def apply_query_runtime_profile(req: Any, modality: str, image_input: str | None) -> Dict[str, Any]:
    """Derive execution knobs for one request without mutating the incoming request object."""
    workload = classify_query_workload(req, modality=modality, image_input=image_input)
    perf_mode_enabled = str(settings.get("ROUTER_PERF_MODE", "0")).strip() == "1"
    simple_query_cap = int(settings.get("ROUTER_SIMPLE_QUERY_MAX_TOKENS", settings.MAX_TOKENS_DEFAULT))
    rag_simple_bypass = str(settings.get("RAG_SIMPLE_QUERY_BYPASS_ENABLED", "1")).strip() == "1"

    use_rag = bool(req.enable_rag_for_answer or req.enable_rag_for_image)
    effective_max_tokens = req.max_tokens or settings.MAX_TOKENS_DEFAULT

    if perf_mode_enabled and workload == "simple_text":
        effective_max_tokens = min(int(effective_max_tokens), max(32, simple_query_cap))

    if workload == "simple_text" and rag_simple_bypass:
        use_rag = False

    if modality in {"vision", "multimodal"} or image_input:
        use_rag = bool(req.enable_rag_for_answer or req.enable_rag_for_image)

    return {
        "workload_class": workload,
        "perf_mode_enabled": perf_mode_enabled,
        "use_rag": use_rag,
        "max_tokens": effective_max_tokens,
    }


async def process_query_request(req: Any) -> Dict[str, Any]:
    """Process one query request with governance, guardrails, and experimentation hooks."""
    if not req or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query obrigatória.")

    input_decision = check_input_guardrails(req.query)
    if not input_decision.allowed:
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
                "reasons": input_decision.reasons,
            },
        )

    pre_budget = check_tenant_budget(req.tenant_id)
    if not pre_budget.allowed:
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
                "reason": pre_budget.reason,
                "daily_spent": pre_budget.daily_spent,
                "monthly_spent": pre_budget.monthly_spent,
                "daily_limit": pre_budget.daily_limit,
                "monthly_limit": pre_budget.monthly_limit,
            },
        )

    modality = (req.modality or "text").lower()
    image_input = req.image_b64
    if not image_input and req.images and len(req.images) > 0:
        image_input = req.images[0]
    if image_input and modality == "text":
        modality = "vision"

    runtime_profile = apply_query_runtime_profile(req, modality=modality, image_input=image_input)

    selected_policy = req.policy_version
    active_policy = get_active_policy()
    if not selected_policy and active_policy:
        selected_policy = active_policy.get("version")
    if active_policy and active_policy.get("version"):
        POLICY_VERSION_ACTIVE.labels(policy_version=str(active_policy["version"])).set(1)

    assigned_variant = None
    if req.experiment_id and settings.AB_TESTING_ENABLED:
        try:
            manager = get_ab_test_manager()
            assignment = manager.get_assignment(
                req.experiment_id,
                req.user_key or req.tenant_id or f"anon:{hash(req.query)}",
            )
            if assignment:
                variant_name, variant_cfg = assignment
                assigned_variant = {"name": variant_name, "config": variant_cfg}
                selected_policy = variant_cfg.get("policy_version", selected_policy)
        except Exception as exc:
            logger.warning(f"[query] Failed experiment assignment: {exc}")

    logger.info(
        "[query] '%s...' (mod=%s, tenant=%s, policy=%s, exp=%s)",
        req.query[:60],
        modality,
        req.tenant_id or "-",
        selected_policy or "-",
        req.experiment_id or "-",
    )

    try:
        result = await route_and_answer(
            query=req.query,
            system_prompt=req.system_prompt or "",
            use_rag=runtime_profile["use_rag"],
            max_tokens=runtime_profile["max_tokens"],
            temperature=req.temperature or settings.TEMPERATURE_DEFAULT,
            modality=modality,
            image_b64=image_input,
            rag_modality=(req.rag_modality or "text").lower(),
            use_cache=req.use_cache,
            timeout_seconds=req.timeout_seconds,
        )
    except asyncio.TimeoutError:
        ROUTER_QUERY_OUTCOME.labels(
            outcome="provider_timeout",
            model="unknown",
            modality=modality,
        ).inc()
        error_info = log_error(
            asyncio.TimeoutError("Request timed out"),
            category=ErrorCategory.PROVIDER_TIMEOUT,
        )
        raise HTTPException(status_code=504, detail=create_error_response(error_info))
    except ProviderCircuitOpenError as exc:
        ROUTER_QUERY_OUTCOME.labels(
            outcome="provider_unavailable",
            model=exc.model or "unknown",
            modality=modality,
        ).inc()
        error_info = log_error(exc, category=ErrorCategory.CIRCUIT_OPEN, model=exc.model)
        raise HTTPException(status_code=503, detail=create_error_response(error_info))
    except ProviderCallError as exc:
        ROUTER_QUERY_OUTCOME.labels(
            outcome=exc.category,
            model=exc.model or "unknown",
            modality=modality,
        ).inc()
        category_map = {
            "provider_timeout": ErrorCategory.PROVIDER_TIMEOUT,
            "provider_rate_limit": ErrorCategory.PROVIDER_RATE_LIMIT,
            "provider_unavailable": ErrorCategory.PROVIDER_UNAVAILABLE,
        }
        category = category_map.get(exc.category, ErrorCategory.PROVIDER_UNAVAILABLE)
        status_code = 504 if category == ErrorCategory.PROVIDER_TIMEOUT else (429 if category == ErrorCategory.PROVIDER_RATE_LIMIT else 502)
        error_info = log_error(exc, category=category, model=exc.model)
        raise HTTPException(status_code=status_code, detail=create_error_response(error_info))
    except Exception as exc:
        ROUTER_QUERY_OUTCOME.labels(
            outcome="provider_unavailable",
            model="unknown",
            modality=modality,
        ).inc()
        error_info = log_error(exc)
        logger.exception(f"[router] Erro: {exc}")
        raise HTTPException(status_code=500, detail=create_error_response(error_info))

    answer_clean, output_guardrail_tags = sanitize_output_guardrails(result.get("answer", ""))
    result["answer"] = answer_clean
    metadata = result.setdefault("metadata", {})
    metadata["guardrail_output_tags"] = output_guardrail_tags
    metadata["policy_version"] = selected_policy
    metadata["experiment_id"] = req.experiment_id
    metadata["experiment_variant"] = assigned_variant
    metadata["tenant_id"] = req.tenant_id
    metadata["workload_class"] = runtime_profile["workload_class"]
    metadata["perf_mode_enabled"] = runtime_profile["perf_mode_enabled"]

    chosen_model = result.get("model", "unknown")
    if selected_policy:
        QUERY_POLICY_APPLIED.labels(policy_version=str(selected_policy)).inc()

    fallback_used = bool(((result.get("route") or {}).get("fallback") or {}).get("used"))
    answer_clean_str = str(result.get("answer", "") or "").strip()
    if chosen_model == "semantic_cache":
        outcome = "cache_hit"
    elif not answer_clean_str:
        outcome = "empty_answer"
    elif fallback_used:
        outcome = "fallback_success"
    else:
        outcome = "success"

    ROUTER_QUERY_OUTCOME.labels(
        outcome=outcome,
        model=chosen_model,
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
    cost_usd = result.get("cost_per_1k", 0)
    metadata = result.get("metadata", {})
    prompt_tokens = metadata.get("prompt_tokens", 0)
    completion_tokens = metadata.get("completion_tokens", 0)
    raw_payload_str = metadata.get("raw_payload")
    uncertainty_score = metadata.get("uncertainty_score", 0.5)
    perf_mode_enabled = str(settings.get("ROUTER_PERF_MODE", "0")).strip() == "1"

    combined_payload = {
        "uncertainty_score": uncertainty_score,
        "queue_enqueued_at": time.time(),
    }
    if not perf_mode_enabled and raw_payload_str:
        combined_payload["raw_payload"] = raw_payload_str

    try:
        task_process_feedback.delay(
            query=req.query,
            answer=result["answer"],
            chosen_model=chosen_model,
            modality=result["modality"],
            latency_s=result["latency_s"],
            cost_val=result["cost_per_1k"],
            image_b64=image_input,
            raw_payload=combined_payload,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as exc:
        logger.error(f"[main] Falha ao despachar tarefa Celery: {exc}")

    try:
        record_tenant_usage(
            tenant_id=req.tenant_id,
            cost_usd=float(cost_usd),
            tokens_in=int(prompt_tokens or 0),
            tokens_out=int(completion_tokens or 0),
            requests=1,
        )
    except Exception as exc:
        logger.warning(f"[query] Failed to record tenant usage: {exc}")

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
