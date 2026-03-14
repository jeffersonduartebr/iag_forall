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
    RESPONSE_ABSTAIN_TOTAL,
    RESPONSE_GROUNDED_TOTAL,
    RESPONSE_REVIEW_STATUS,
    RESPONSE_VERIFICATION_STATUS,
    ROUTER_QUERY_OUTCOME,
    logger,
)
from ..providers_async import ProviderCallError, ProviderCircuitOpenError
from ..router_core import route_and_answer
from ..roadmap_features import create_response_review
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
_SIMPLE_QUERY_HINTS = (
    "quanto é",
    "what is",
    "who is",
    "when was",
    "capital of",
    "defina",
    "define",
)
_SOURCE_REQUIRED_HINTS = (
    "fonte",
    "fontes",
    "source",
    "sources",
    "artigo",
    "paper",
    "lei",
    "decreto",
    "resolução",
    "norma",
    "manual",
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp one floating-point value into the expected confidence interval."""
    return max(lower, min(upper, float(value)))


def enrich_result_reliability(result: Dict[str, Any]) -> Dict[str, Any]:
    """Attach confidence, grounding, verification, and review hints to one result."""
    metadata = result.setdefault("metadata", {})
    route = result.setdefault("route", {})
    fallback = (route.get("fallback") or {})
    answer_text = str(result.get("answer", "") or "").strip()
    uncertainty = float(metadata.get("uncertainty_score", 0.5) or 0.5)
    grounded = bool(metadata.get("grounded"))
    retrieval_mode = str(metadata.get("retrieval_mode") or "no_retrieval")
    workload_class = str(metadata.get("workload_class") or "reasoning")
    output_guardrail_tags = metadata.get("guardrail_output_tags") or []
    fallback_used = bool(fallback.get("used"))

    confidence_score = 1.0 - uncertainty
    if grounded:
        confidence_score += 0.20
    if retrieval_mode != "no_retrieval" and not grounded:
        confidence_score -= 0.20
    if fallback_used:
        confidence_score -= 0.10
    if output_guardrail_tags:
        confidence_score -= 0.10
    if not answer_text:
        confidence_score -= 0.60
    confidence_score = round(_clamp(confidence_score), 3)

    if confidence_score >= 0.75:
        confidence_band = "high"
    elif confidence_score >= 0.45:
        confidence_band = "medium"
    else:
        confidence_band = "low"

    if not answer_text:
        verification_status = "unsupported"
    elif grounded and confidence_score >= 0.70:
        verification_status = "supported"
    elif grounded or confidence_score >= 0.45:
        verification_status = "weakly_supported"
    else:
        verification_status = "unsupported"

    abstained = False
    abstain_reason = None
    if not answer_text:
        abstained = True
        abstain_reason = "empty_answer"
    elif confidence_band == "low" and workload_class in {"knowledge_lookup", "reasoning"} and verification_status == "unsupported":
        abstained = True
        abstain_reason = "low_confidence"
    elif retrieval_mode != "no_retrieval" and workload_class == "knowledge_lookup" and not grounded and confidence_score < 0.55:
        abstained = True
        abstain_reason = "insufficient_evidence"

    if abstained:
        safe_answer = (
            "Nao tenho evidencia suficiente para responder com confianca. "
            "Tente reformular a pergunta ou fornecer mais contexto."
        )
        result["answer"] = safe_answer
        answer_text = safe_answer

    review_status = "needs_review" if abstained or verification_status == "unsupported" or confidence_band == "low" else "auto_approved"

    metadata["confidence_score"] = confidence_score
    metadata["confidence_band"] = confidence_band
    metadata["abstained"] = abstained
    metadata["abstain_reason"] = abstain_reason
    metadata["verification_status"] = verification_status
    metadata["review_status"] = review_status
    metadata["grounded"] = grounded
    metadata["citations"] = list(metadata.get("citations") or [])
    metadata["evidence_snippets"] = list(metadata.get("evidence_snippets") or [])
    result["confidence_score"] = confidence_score
    result["confidence_band"] = confidence_band
    result["abstained"] = abstained
    result["abstain_reason"] = abstain_reason
    result["grounded"] = grounded
    result["citations"] = metadata["citations"]
    result["evidence_snippets"] = metadata["evidence_snippets"]
    result["knowledge_version"] = metadata.get("knowledge_version")
    result["verification_status"] = verification_status
    result["review_status"] = review_status
    try:
        RESPONSE_GROUNDED_TOTAL.labels(grounded="true" if grounded else "false").inc()
        RESPONSE_VERIFICATION_STATUS.labels(status=verification_status).inc()
        RESPONSE_REVIEW_STATUS.labels(status=review_status).inc()
        if abstained:
            RESPONSE_ABSTAIN_TOTAL.labels(reason=str(abstain_reason or "unknown")).inc()
    except Exception:
        pass
    return result


def classify_query_workload(req: Any, modality: str, image_input: str | None) -> str:
    """Classify a request into a small set of performance-relevant workload classes."""
    if image_input or modality in {"vision", "multimodal"}:
        return "vision"

    if str(settings.get("ROUTER_QUERY_CLASSIFIER_ENABLED", "1")).strip() != "1":
        return "reasoning"

    query = str(getattr(req, "query", "") or "").strip()
    lowered = query.lower()

    if any(hint in lowered for hint in _RETRIEVAL_HINTS):
        return "knowledge_lookup"
    if any(hint in lowered for hint in _REASONING_HINTS):
        return "reasoning"

    token_count = len(re.findall(r"\w+", query))
    if any(hint in lowered for hint in _SIMPLE_QUERY_HINTS):
        return "simple_text"
    if token_count <= 18 and len(query) <= 140 and "?" in query:
        return "simple_text"
    if token_count <= 12 and len(query) <= 100:
        return "simple_text"
    return "reasoning"


def _infer_retrieval_profile(query: str, workload: str) -> str:
    """Choose the cheapest retrieval depth that still matches the query intent."""
    lowered = str(query or "").lower()
    if workload == "simple_text":
        return "no_retrieval"
    if workload == "knowledge_lookup":
        if any(hint in lowered for hint in _SOURCE_REQUIRED_HINTS):
            return "full_retrieval"
        return "light_retrieval"
    if workload == "vision":
        return "full_retrieval"
    return "full_retrieval"


def apply_query_runtime_profile(req: Any, modality: str, image_input: str | None) -> Dict[str, Any]:
    """Derive execution knobs for one request without mutating the incoming request object."""
    workload = classify_query_workload(req, modality=modality, image_input=image_input)
    perf_mode_enabled = str(settings.get("ROUTER_PERF_MODE", "0")).strip() == "1"
    simple_query_cap = int(settings.get("ROUTER_SIMPLE_QUERY_MAX_TOKENS", settings.MAX_TOKENS_DEFAULT))
    simple_text_max_fallbacks = max(1, int(settings.get("ROUTER_SIMPLE_TEXT_MAX_FALLBACKS", 1)))
    rag_simple_bypass = str(settings.get("RAG_SIMPLE_QUERY_BYPASS_ENABLED", "1")).strip() == "1"
    light_top_k = max(1, int(settings.get("RAG_LIGHT_TOP_K", 2)))
    light_context_budget = max(128, int(settings.get("RAG_LIGHT_CONTEXT_TOKEN_BUDGET", 480)))
    full_context_budget = max(light_context_budget, int(settings.get("RAG_FULL_CONTEXT_TOKEN_BUDGET", settings.get("RAG_CONTEXT_TOKEN_BUDGET", 1200))))
    rerank_for_light = str(settings.get("RERANK_ENABLED_FOR_LIGHT_RETRIEVAL", "0")).strip() == "1"

    use_rag = bool(req.enable_rag_for_answer or req.enable_rag_for_image)
    effective_max_tokens = req.max_tokens or settings.MAX_TOKENS_DEFAULT
    retrieval_mode = _infer_retrieval_profile(getattr(req, "query", ""), workload)
    top_k = max(1, light_top_k)
    context_token_budget = full_context_budget
    rerank_enabled = True
    max_fallbacks = 2
    needs_retrieval = retrieval_mode != "no_retrieval"
    needs_rerank = True
    interactive_priority = "normal"

    if perf_mode_enabled and workload == "simple_text":
        effective_max_tokens = min(int(effective_max_tokens), max(32, simple_query_cap))

    if workload == "simple_text" and rag_simple_bypass:
        use_rag = False
        retrieval_mode = "no_retrieval"
        top_k = 1
        context_token_budget = light_context_budget
        rerank_enabled = False
        needs_retrieval = False
        needs_rerank = False
        interactive_priority = "high"
        max_fallbacks = simple_text_max_fallbacks
    elif workload == "knowledge_lookup":
        use_rag = True if use_rag or perf_mode_enabled else use_rag
        retrieval_mode = _infer_retrieval_profile(getattr(req, "query", ""), workload)
        top_k = light_top_k
        context_token_budget = light_context_budget if retrieval_mode == "light_retrieval" else full_context_budget
        rerank_enabled = rerank_for_light if retrieval_mode == "light_retrieval" else True
        needs_retrieval = True
        needs_rerank = bool(rerank_enabled)
        interactive_priority = "high"
        max_fallbacks = 1 if perf_mode_enabled else 2
    elif workload == "reasoning":
        retrieval_mode = "full_retrieval" if use_rag else "no_retrieval"
        top_k = max(3, light_top_k + 1)
        context_token_budget = full_context_budget
        rerank_enabled = True
        needs_retrieval = retrieval_mode != "no_retrieval"
        needs_rerank = True
        interactive_priority = "normal"
        max_fallbacks = 2

    if modality in {"vision", "multimodal"} or image_input:
        use_rag = bool(req.enable_rag_for_answer or req.enable_rag_for_image)
        retrieval_mode = "full_retrieval" if use_rag else "no_retrieval"
        top_k = max(3, light_top_k + 1)
        context_token_budget = full_context_budget
        rerank_enabled = True
        needs_retrieval = retrieval_mode != "no_retrieval"
        needs_rerank = True
        interactive_priority = "high"
        max_fallbacks = 2

    return {
        "workload_class": workload,
        "perf_mode_enabled": perf_mode_enabled,
        "use_rag": use_rag,
        "max_tokens": effective_max_tokens,
        "runtime_hints": {
            "workload_class": workload,
            "retrieval_mode": retrieval_mode if use_rag else "no_retrieval",
            "rag_top_k": top_k,
            "rag_context_token_budget": context_token_budget,
            "rag_rerank_enabled": rerank_enabled,
            "max_fallbacks": max_fallbacks,
            "needs_retrieval": bool(use_rag and needs_retrieval),
            "needs_rerank": bool(use_rag and needs_rerank),
            "interactive_priority": interactive_priority,
        },
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
            runtime_hints=runtime_profile["runtime_hints"],
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
    metadata["retrieval_mode"] = runtime_profile["runtime_hints"]["retrieval_mode"]
    result = enrich_result_reliability(result)

    chosen_model = result.get("model", "unknown")
    if selected_policy:
        QUERY_POLICY_APPLIED.labels(policy_version=str(selected_policy)).inc()

    fallback_used = bool(((result.get("route") or {}).get("fallback") or {}).get("used"))
    answer_clean_str = str(result.get("answer", "") or "").strip()
    if chosen_model == "semantic_cache":
        outcome = "cache_hit"
    elif bool(result.get("abstained")) and str(result.get("abstain_reason") or "") == "empty_answer":
        outcome = "empty_answer"
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
    cost_usd = result.get("estimated_cost_usd", result.get("cost_per_1k", 0))
    metadata = result.get("metadata", {})
    prompt_tokens = metadata.get("prompt_tokens", 0)
    completion_tokens = metadata.get("completion_tokens", 0)
    raw_payload_str = metadata.get("raw_payload")
    uncertainty_score = metadata.get("uncertainty_score", 0.5)
    perf_mode_enabled = str(settings.get("ROUTER_PERF_MODE", "0")).strip() == "1"
    confidence_score = metadata.get("confidence_score")
    confidence_band = metadata.get("confidence_band")
    grounded = bool(metadata.get("grounded"))
    abstained = bool(metadata.get("abstained"))
    abstain_reason = metadata.get("abstain_reason")
    verification_status = metadata.get("verification_status")
    knowledge_version = metadata.get("knowledge_version")
    review_status = metadata.get("review_status")
    citations = metadata.get("citations") or []
    evidence_snippets = metadata.get("evidence_snippets") or []

    combined_payload = {
        "uncertainty_score": uncertainty_score,
        "queue_enqueued_at": time.time(),
        "confidence_score": confidence_score,
        "confidence_band": confidence_band,
        "grounded": grounded,
        "abstained": abstained,
        "abstain_reason": abstain_reason,
        "verification_status": verification_status,
        "knowledge_version": knowledge_version,
        "review_status": review_status,
        "citations": citations,
        "evidence_snippets": evidence_snippets,
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
            cost_val=result.get("estimated_cost_usd", result.get("cost_per_1k", 0.0)),
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

    if review_status == "needs_review":
        try:
            create_response_review(
                correlation_id=metadata.get("correlation_id"),
                tenant_id=req.tenant_id,
                query_text=req.query,
                answer=result["answer"],
                chosen_model=chosen_model,
                confidence_score=float(confidence_score) if confidence_score is not None else None,
                confidence_band=str(confidence_band) if confidence_band else None,
                grounded=grounded,
                verification_status=str(verification_status) if verification_status else None,
                review_reason=str(abstain_reason or verification_status or "low_confidence"),
                metadata={
                    "citations": citations,
                    "evidence_snippets": evidence_snippets,
                    "knowledge_version": knowledge_version,
                    "workload_class": metadata.get("workload_class"),
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
