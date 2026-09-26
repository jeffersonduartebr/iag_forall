# Objective: Every served request leaves a query_log row, including the ones outside the feedback loop.
"""The query_log row for requests that skip the judges/bandit loop.

The feedback worker writes the row of an ordinary request. Three kinds of request never reach it: instrument calls
with ``pinned_model`` (not a policy decision, so nothing may learn from them), tool-call turns (no text to judge)
and requests whose feedback task could not be enqueued (broker down). Their row is written here, directly, with
``quality`` NULL and ``quality_source`` naming why; nothing is judged and nothing learns from it.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def parametros(req: Any) -> Dict[str, Any]:
    """What the request asked for. The system prompt goes as a hash: it carries the private answer key."""
    sistema = getattr(req, "system_prompt", None) or ""
    return {
        "temperature": getattr(req, "temperature", None),
        "max_tokens": getattr(req, "max_tokens", None),
        "system_prompt_sha256": hashlib.sha256(sistema.encode()).hexdigest() if sistema else None,
        "system_prompt_chars": len(sistema),
        "enable_rag_for_answer": getattr(req, "enable_rag_for_answer", None),
        "rag_modality": getattr(req, "rag_modality", None),
        "rag_filter": getattr(req, "rag_filter", None),
        "images": len(getattr(req, "images", None) or []) + (1 if getattr(req, "image_b64", None) else 0),
        "pinned_model": getattr(req, "pinned_model", None),
        "use_cache": getattr(req, "use_cache", None),
    }


def gravar_sem_feedback(req: Any, result: Dict[str, Any], payload: Dict[str, Any], image_input: Optional[str],
                        motivo: str) -> None:
    """Write the row now; a failure is logged loudly and never breaks the response already produced."""
    from ..query_service import insert_query_log
    from .feedback_stages import _reliability_fields, _research_fields  # carrega log_fields sem ciclo

    try:
        insert_query_log(
            query_text=req.query,
            model=result["model"],
            modality=result["modality"],
            image_provided=bool(image_input),
            answer=result.get("answer") or "",
            image_output_b64=None,
            latency_s=float(result.get("latency_s") or 0.0),
            estimated_cost_usd=float(result.get("estimated_cost_usd", result.get("cost_per_1k", 0.0)) or 0.0),
            quality=None,
            reward=None,
            quality_source=motivo,
            decision=payload.get("decision") or None,
            correlation_id=payload.get("correlation_id"),
            detected_complexity=payload.get("detected_complexity"),
            context_label=motivo,
            tenant_id=payload.get("tenant_id"),
            raw_payload=payload,
            **_reliability_fields(payload),
            **_research_fields(payload),
        )
    except Exception as exc:
        logger.error("[registro] query_log NÃO foi escrito (%s, %s): %s", motivo, payload.get("correlation_id"), exc)


def despachar(req: Any, result: Dict[str, Any], image_input: Optional[str], payload: Dict[str, Any],
              is_tool_turn: bool) -> bool:
    """Enqueue the feedback loop, or write the row directly. Returns True when the loop was enqueued."""
    from ..tasks import task_process_feedback

    if getattr(req, "pinned_model", None):
        gravar_sem_feedback(req, result, payload, image_input, "pinned_model")
        return False
    if is_tool_turn:
        gravar_sem_feedback(req, result, payload, image_input, "tool_turn")
        return False
    metadata = result.get("metadata") or {}
    try:
        task_process_feedback.delay(
            query=req.query,
            answer=result["answer"],
            chosen_model=result["model"],
            modality=result["modality"],
            latency_s=result["latency_s"],
            cost_val=result.get("estimated_cost_usd", result.get("cost_per_1k", 0.0)),
            image_b64=image_input,
            raw_payload=payload,
            prompt_tokens=metadata.get("prompt_tokens", 0),
            completion_tokens=metadata.get("completion_tokens", 0),
        )
        return True
    except Exception as exc:
        logger.error(f"[main] Falha ao despachar tarefa Celery: {exc}")
        gravar_sem_feedback(req, result, payload, image_input, "enqueue_failed")
        return False
