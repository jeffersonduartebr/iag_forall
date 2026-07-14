# -*- coding: utf-8 -*-
# Objective: Build public query-response payloads from processed router results.
"""Translate processed query results into the public API response contract."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..schemas import (
    CandidateResult,
    ConfidenceBand,
    EvidenceSnippet,
    QueryResponse,
    ResponseCitation,
    ResponseDiagnostics,
    ResponseProvenance,
    ReviewStatus,
    RouteDecision,
    VerificationStatus,
)


def safe_parse_json(payload: Any) -> Any:
    """Decode JSON payloads when they arrive as strings."""
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return payload
    return payload


def build_query_response(result: Dict[str, Any], correlation_id: str | None) -> QueryResponse:
    """Build the public response model from one processed router result."""
    metadata = result.get("metadata", {}) or {}
    raw_payload_str = metadata.get("raw_payload")
    prompt_tokens = metadata.get("prompt_tokens", 0)
    completion_tokens = metadata.get("completion_tokens", 0)
    parsed_payload = safe_parse_json(raw_payload_str)
    route_raw = result.get("route", {}) or {}
    candidates_raw = result.get("candidates", []) or []

    diagnostics = ResponseDiagnostics(
        route=RouteDecision(**route_raw) if route_raw else None,
        candidates=[CandidateResult(**candidate) for candidate in candidates_raw],
        payload=parsed_payload,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        uncertainty_score=metadata.get("uncertainty_score"),
        guardrail_output_tags=list(metadata.get("guardrail_output_tags") or []),
        experiment_id=metadata.get("experiment_id"),
        experiment_variant=metadata.get("experiment_variant"),
        tenant_id=metadata.get("tenant_id"),
        perf_mode_enabled=metadata.get("perf_mode_enabled"),
        retrieval_mode=metadata.get("retrieval_mode"),
        workload_class=metadata.get("workload_class"),
        stage_timings_ms=metadata.get("stage_timings_ms"),
        raw_payload=parsed_payload,
    )

    return QueryResponse(
        answer=result["answer"],
        model=result["model"],
        modality=result["modality"],
        image_output_b64=result.get("image_output_b64"),
        correlation_id=correlation_id,
        estimated_cost_usd=result.get("estimated_cost_usd", result.get("cost_per_1k", 0.0)),
        confidence_score=result.get("confidence_score"),
        confidence_band=ConfidenceBand(result["confidence_band"]) if result.get("confidence_band") else None,
        abstained=bool(result.get("abstained")),
        abstain_reason=result.get("abstain_reason"),
        verification_status=VerificationStatus(result["verification_status"]) if result.get("verification_status") else None,
        review_status=ReviewStatus(result["review_status"]) if result.get("review_status") else None,
        provenance=ResponseProvenance(
            grounded=bool(result.get("grounded")),
            citations=[ResponseCitation(**item) for item in (result.get("citations") or [])],
            evidence_snippets=[EvidenceSnippet(**item) for item in (result.get("evidence_snippets") or [])],
            knowledge_version=result.get("knowledge_version"),
        ),
        tool_calls=result.get("tool_calls"),
        finish_reason=result.get("finish_reason"),
        diagnostics=diagnostics,
    )
