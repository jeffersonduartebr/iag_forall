# -*- coding: utf-8 -*-
# Objective: Reliability annotations (confidence, grounding, verification, review) for query results.
"""Attach confidence, grounding, verification and review hints to one router result.

Extracted from ``services.query_runtime`` (re-exported there).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..observability import (
    RESPONSE_ABSTAIN_TOTAL,
    RESPONSE_GROUNDED_TOTAL,
    RESPONSE_REVIEW_STATUS,
    RESPONSE_VERIFICATION_STATUS,
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp one floating-point value into the expected confidence interval."""
    return max(lower, min(upper, float(value)))


SAFE_ABSTAIN_ANSWER = (
    "Nao tenho evidencia suficiente para responder com confianca. "
    "Tente reformular a pergunta ou fornecer mais contexto."
)
_TOOL_TURN_FIELDS = {
    "confidence_score": None,
    "confidence_band": None,
    "abstained": False,
    "abstain_reason": None,
    "verification_status": None,
    "review_status": "auto_approved",
}


def confidence_score(
    uncertainty: float, *, answered: bool, grounded: bool, retrieval_used: bool, fallback_used: bool, flagged: bool
) -> float:
    """1 - u(q), adjusted by grounding, retrieval without evidence, fallback, output flags and empty answers."""
    score = 1.0 - uncertainty
    score += 0.20 if grounded else 0.0
    score -= 0.20 if retrieval_used and not grounded else 0.0
    score -= 0.10 if fallback_used else 0.0
    score -= 0.10 if flagged else 0.0
    score -= 0.60 if not answered else 0.0
    return round(_clamp(score), 3)


def confidence_band(score: float) -> str:
    if score >= 0.75:
        return "high"
    return "medium" if score >= 0.45 else "low"


def verification_status(*, answered: bool, grounded: bool, score: float) -> str:
    if not answered:
        return "unsupported"
    if grounded and score >= 0.70:
        return "supported"
    return "weakly_supported" if grounded or score >= 0.45 else "unsupported"


def abstain_reason(
    *,
    answered: bool,
    band: str,
    verification: str,
    workload_class: str,
    complexity: str,
    retrieval_used: bool,
    grounded: bool,
    score: float,
) -> Optional[str]:
    """Why the answer should be replaced by a safe abstention (``None`` keeps it)."""
    if not answered:
        return "empty_answer"
    weak = band == "low" and verification == "unsupported"
    if weak and (workload_class in {"knowledge_lookup", "reasoning"} or complexity in {"high", "expert"}):
        return "low_confidence"
    if retrieval_used and workload_class == "knowledge_lookup" and not grounded and score < 0.55:
        return "insufficient_evidence"
    return None


def _publish(result: Dict[str, Any], metadata: Dict[str, Any], fields: Dict[str, Any]) -> None:
    metadata.update(fields)
    result.update(fields)


def _provenance_fields(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "grounded": bool(metadata.get("grounded")),
        "citations": list(metadata.get("citations") or []),
        "evidence_snippets": list(metadata.get("evidence_snippets") or []),
    }


def _count_outcome(grounded: bool, verification: str, review: str, reason: Optional[str]) -> None:
    try:
        RESPONSE_GROUNDED_TOTAL.labels(grounded="true" if grounded else "false").inc()
        RESPONSE_VERIFICATION_STATUS.labels(status=verification).inc()
        RESPONSE_REVIEW_STATUS.labels(status=review).inc()
        if reason is not None:
            RESPONSE_ABSTAIN_TOTAL.labels(reason=str(reason or "unknown")).inc()
    except Exception:
        pass


def enrich_result_reliability(result: Dict[str, Any]) -> Dict[str, Any]:
    """Attach confidence, grounding, verification, and review hints to one result."""
    metadata = result.setdefault("metadata", {})
    route = result.setdefault("route", {})

    # Turno de tool call: o modelo devolveu tool_calls (texto vazio é esperado).
    # Não aplicar lógica de abstenção/verificação de resposta vazia.
    if result.get("finish_reason") == "tool_calls" or result.get("tool_calls"):
        metadata.update(_TOOL_TURN_FIELDS)
        result.update(_TOOL_TURN_FIELDS)
        result.update(_provenance_fields(metadata))
        result["knowledge_version"] = metadata.get("knowledge_version")
        return result

    answered = bool(str(result.get("answer", "") or "").strip())
    grounded = bool(metadata.get("grounded"))
    retrieval_used = str(metadata.get("retrieval_mode") or "no_retrieval") != "no_retrieval"
    workload_class = str(metadata.get("workload_class") or "reasoning")
    score = confidence_score(
        float(metadata.get("uncertainty_score", 0.5) or 0.5),
        answered=answered,
        grounded=grounded,
        retrieval_used=retrieval_used,
        fallback_used=bool((route.get("fallback") or {}).get("used")),
        flagged=bool(metadata.get("guardrail_output_tags")),
    )
    band = confidence_band(score)
    verification = verification_status(answered=answered, grounded=grounded, score=score)
    reason = abstain_reason(
        answered=answered,
        band=band,
        verification=verification,
        workload_class=workload_class,
        complexity=str(metadata.get("detected_complexity") or ""),
        retrieval_used=retrieval_used,
        grounded=grounded,
        score=score,
    )
    if reason is not None:
        result["answer"] = SAFE_ABSTAIN_ANSWER
    review = "needs_review" if reason is not None or verification == "unsupported" or band == "low" else "auto_approved"

    _publish(result, metadata, _provenance_fields(metadata))
    _publish(
        result,
        metadata,
        {
            "confidence_score": score,
            "confidence_band": band,
            "abstained": reason is not None,
            "abstain_reason": reason,
            "verification_status": verification,
            "review_status": review,
        },
    )
    result["knowledge_version"] = metadata.get("knowledge_version")
    _count_outcome(grounded, verification, review, reason)
    return result
