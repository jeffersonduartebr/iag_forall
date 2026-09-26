# Objective: Build the payload the async feedback loop persists for one query.
"""The bridge between a served response and the row that records it.

Everything the feedback worker knows about a request arrives through this
dictionary, which is why it is worth keeping in one readable place: a key
missing here is a column silently left NULL in ``query_log``, and that is how
``candidates`` and ``pareto_front`` came to be written as empty lists for the
whole life of the system.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

#: Campos copiados de `metadata` tal e qual. A lista é explícita para que
#: acrescentar um campo à resposta e esquecer de o persistir seja visível.
_PASSTHROUGH = (
    "detected_complexity",
    "workload_class",
    "confidence_score",
    "confidence_band",
    "abstain_reason",
    "verification_status",
    "knowledge_version",
    "review_status",
)


#: O rastro da execução que não tem coluna própria: vai inteiro para `query_log.trace_json`.
_TRACE_METADATA = (
    "stage_timings_ms",
    "retrieval_mode",
    "retrieval_skipped_reason",
    "policy_version",
    "experiment_id",
    "experiment_variant",
    "guardrail_output_tags",
    "answer_before_abstention",
    "cash_cost_usd",
    "imputed_cost_usd",
    "load_time",
)


def _trace(result: Dict[str, Any], metadata: Dict[str, Any], route_path: Optional[str]) -> Dict[str, Any]:
    """Everything needed to reconstruct how the answer was produced, beyond the dedicated columns."""
    trace = {key: metadata.get(key) for key in _TRACE_METADATA if metadata.get(key) is not None}
    trace["fallback"] = (result.get("route") or {}).get("fallback") or {}
    trace["tool_calls"] = result.get("tool_calls") or None
    trace["route_path"] = route_path
    trace["cached"] = bool(metadata.get("cached"))
    return trace


def build_feedback_payload(
    result: Dict[str, Any],
    *,
    tenant_id: Optional[str],
    include_raw: bool,
    participant: Optional[str] = None,
    episode_id: Optional[str] = None,
    route_path: Optional[str] = None,
    request_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble what the feedback loop needs to persist this query."""
    metadata = result.get("metadata", {}) or {}
    payload: Dict[str, Any] = {
        "uncertainty_score": metadata.get("uncertainty_score", 0.5),
        "queue_enqueued_at": time.time(),
        "tenant_id": tenant_id,
        "grounded": bool(metadata.get("grounded")),
        "abstained": bool(metadata.get("abstained")),
        "citations": metadata.get("citations") or [],
        "evidence_snippets": metadata.get("evidence_snippets") or [],
        "openrouter_exploration": bool(metadata.get("openrouter_exploration")),
        "exploration_info": metadata.get("exploration_info") or {},
        # O registo auditável da decisão: candidatos, objectivos de cada um,
        # frente de Pareto e os pesos NSGA-II em vigor. Sem os pesos o score
        # escalarizado não é reinterpretável, porque o updater reescreve-os.
        "decision": result.get("decision") or {},
        # Liga a linha de query_log ao rasto de execução estruturado. Estava a
        # ser gerado e nunca chegava à tabela, por isso não havia como juntar
        # uma linha ao rasto que a produziu.
        "correlation_id": metadata.get("correlation_id"),
        # Pesquisa (Caso 1): quem perguntou (código pseudônimo), em que episódio, com quantos tokens.
        "participant": participant,
        "episode_id": episode_id,
        "prompt_tokens": int(metadata.get("prompt_tokens") or 0),
        "completion_tokens": int(metadata.get("completion_tokens") or 0),
        "reasoning_tokens": int(metadata.get("reasoning_tokens") or 0),
        "finish_reason": result.get("finish_reason"),
        "trace": {**_trace(result, metadata, route_path), "request": request_params or {}},
    }
    payload.update({key: metadata.get(key) for key in _PASSTHROUGH})
    if include_raw and metadata.get("raw_payload"):
        payload["raw_payload"] = metadata["raw_payload"]
    return payload
