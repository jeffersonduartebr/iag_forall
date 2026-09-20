# Objective: Build the query_log columns for one feedback row.
"""The mapping from a served request onto the columns that record it.

Split out of ``feedback_stages`` so the persistence contract is in one readable
place. A key missing here is a column silently left NULL — which is how
``decision_json`` came to be empty for the whole life of the system before it
existed, and how ``candidates`` was written as an empty list before that.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .feedback_stages import FeedbackRequest, Quality


def _optional_str(value: Any) -> Optional[str]:
    return str(value) if value else None


def _reliability_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    confidence = payload.get("confidence_score")
    return {
        "confidence_score": float(confidence) if confidence is not None else None,
        "confidence_band": _optional_str(payload.get("confidence_band")),
        "abstained": bool(payload.get("abstained")),
        "abstain_reason": _optional_str(payload.get("abstain_reason")),
        "grounded": bool(payload.get("grounded")),
        "verification_status": _optional_str(payload.get("verification_status")),
        "knowledge_version": _optional_str(payload.get("knowledge_version")),
        "review_status": _optional_str(payload.get("review_status")),
    }


def _formative_fields(quality: Quality, fb: FeedbackRequest) -> Dict[str, Any]:
    """The formative columns for one query_log row.

    ``quality_semantics`` records which meaning ``quality`` carries on this row,
    so a later analysis never has to guess whether a number came from the rubric
    mean or from the calibrated score.
    """
    from .quality_semantics import current_semantics

    rubric = quality.judge_rubric or {}
    return {
        # Auditoria da decisão: os candidatos e a frente de Pareto tal como
        # existiam no momento da escolha. `{}` só quando não houve comparação
        # (cache hit, modo de emergência) — nunca por se ter deitado fora.
        "decision": fb.payload.get("decision") or None,
        "correlation_id": fb.payload.get("correlation_id"),
        "quality_semantics": current_semantics(),
        "q_tech": rubric.get("q_tech"),
        "q_calibrado": rubric.get("q_calibrado"),
        "p_entrega": rubric.get("p_entrega"),
        "detected_complexity": fb.payload.get("detected_complexity"),
    }
