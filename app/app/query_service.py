# -*- coding: utf-8 -*-
# Objective: Application runtime code for query service.
"""Persist query execution records for analysis, feedback, and offline review.

The query log stores the user request, chosen model, answer payload, optional
multimodal artifacts, embeddings, and summary quality/cost metadata. The
schema is intentionally denormalized so offline analysis and research workflows
can inspect a single record without reconstructing context from multiple tables.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db import get_engine

# ============================================================
# Logging
# ============================================================
logger = logging.getLogger("query_service")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] query_service: %(message)s",
    )


# ============================================================
# DB connection (using centralized engine)
# ============================================================
def _get_engine():
    """Return the shared SQLAlchemy engine managed by the database module."""
    return get_engine()


# Lazy engine accessor for backward compatibility
engine: Any = property(lambda self: _get_engine())


class _EngineProxy:
    """Expose a minimal engine-like interface for legacy callers."""

    def begin(self):
        """Open a transactional connection using the shared engine."""
        return get_engine().begin()

    def connect(self):
        """Open a plain connection using the shared engine."""
        return get_engine().connect()

    def execute(self, *args, **kwargs):
        """Forward direct execution calls to the shared engine."""
        return get_engine().execute(*args, **kwargs)


engine = _EngineProxy()


# ============================================================
# Helpers
# ============================================================

def _to_blob(vec) -> Optional[bytes]:
    """Convert an embedding-like vector into the float32 bytes stored in the BYTEA column."""
    if vec is None:
        return None
    try:
        return np.asarray(vec, dtype=np.float32).tobytes()
    except Exception:
        return None


def _safe_json(obj: dict | list | str | None) -> str:
    """Serialize payload data to JSON while redacting common secret fields."""
    sensitive_keys = {
        "api_key",
        "authorization",
        "token",
        "password",
        "secret",
        "access_token",
        "refresh_token",
    }

    def _redact(value):
        """Recursively redact sensitive keys before JSON serialization."""
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                k = _sem_nul(k)
                if str(k).lower() in sensitive_keys:
                    out[k] = "***REDACTED***"
                else:
                    out[k] = _redact(v)
            return out
        if isinstance(value, list):
            return [_redact(v) for v in value]
        return _sem_nul(value)

    try:
        return json.dumps(_redact(obj), ensure_ascii=False, default=str)
    except Exception:
        return "{}"


# ============================================================
# Semântica de qualidade
# ============================================================

def _current_semantics() -> str:
    """The active quality semantics, resolved defensively.

    A row whose semantics is unknown is unattributable later, so this never
    returns empty: an unreadable setting falls back to the legacy value rather
    than leaving the column to guesswork.
    """
    try:
        from app.services.quality_semantics import current_semantics

        return current_semantics()
    except Exception:
        return "rubric_v1"


# ============================================================
# Inserção multimodal completa (a tabela é criada pelo alembic_pg)
# ============================================================


def _sem_nul(valor: Any) -> Any:
    """PostgreSQL TEXT rejects NUL bytes (MariaDB stored them): strip them from every string that is written."""
    return valor.replace("\x00", "") if isinstance(valor, str) else valor


def insert_query_log(
    *,
    query_text: str,
    model: str,
    modality: str,
    image_provided: bool,
    answer: str,
    image_output_b64: Optional[str],
    latency_s: float,
    estimated_cost_usd: float,
    quality: Optional[float],
    reward: Optional[float],
    quality_source: str = "unknown",
    judge_sampled: bool = False,
    quality_semantics: Optional[str] = None,
    q_tech: Optional[float] = None,
    q_calibrado: Optional[float] = None,
    p_entrega: Optional[float] = None,
    detected_complexity: Optional[str] = None,
    decision: Optional[dict] = None,
    correlation_id: Optional[str] = None,
    predicted_error_prob: Optional[float] = None,
    confidence_score: Optional[float] = None,
    confidence_band: Optional[str] = None,
    abstained: bool = False,
    abstain_reason: Optional[str] = None,
    grounded: bool = False,
    verification_status: Optional[str] = None,
    knowledge_version: Optional[str] = None,
    review_status: Optional[str] = None,
    context_label: Optional[str] = None,
    tenant_id: Optional[str] = None,
    raw_payload: dict | list | str | None = None,
    participant: Optional[str] = None,
    episode_id: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    reasoning_tokens: Optional[int] = None,
    finish_reason: Optional[str] = None,
    trace: Optional[dict] = None,

    # embeddings
    query_embedding: Optional[List[float]] = None,
    answer_embedding: Optional[List[float]] = None,
) -> None:
    """Insert one fully-populated router execution record into `query_log`.

    Callers provide the normalized execution summary plus any optional
    multimodal payloads and embeddings. The function redacts sensitive payload
    fields, strips NUL bytes from text and stores binary vectors in the compact
    representation expected by the schema.
    """
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO query_log
                    (query_text, chosen_model, modality, image_provided,
                     answer, image_output_b64,
                     query_embedding, answer_embedding,
                     quality, quality_source, judge_sampled, predicted_error_prob,
                     confidence_score, confidence_band, abstained, abstain_reason,
                     grounded, verification_status, knowledge_version, review_status,
                     latency_s, estimated_cost_usd, cost_per_1k, reward,
                     quality_semantics, q_tech, q_calibrado, p_entrega, detected_complexity,
                     decision_json, correlation_id,
                     context_label, tenant_id, raw_payload,
                     participant, episode_id, prompt_tokens, completion_tokens, reasoning_tokens,
                     finish_reason, trace_json)
                    VALUES
                     (:q, :m, :mod, :ip,
                     :ans, :img,
                     :qemb, :aemb,
                     :qual, :quality_source, :judge_sampled, :predicted_error_prob,
                     :confidence_score, :confidence_band, :abstained, :abstain_reason,
                     :grounded, :verification_status, :knowledge_version, :review_status,
                     :lat, :estimated_cost_usd, :cost, :rew,
                     :quality_semantics, :q_tech, :q_calibrado, :p_entrega, :detected_complexity,
                     :decision_json, :correlation_id,
                     :ctx, :tenant_id, :payload,
                     :participant, :episode_id, :prompt_tokens, :completion_tokens, :reasoning_tokens,
                     :finish_reason, :trace_json)
                """),
                {chave: _sem_nul(valor) for chave, valor in {
                    "q": query_text,
                    "m": model,
                    "mod": modality,
                    "ip": 1 if image_provided else 0,
                    "ans": answer,
                    "img": image_output_b64,
                    "qemb": _to_blob(query_embedding),
                    "aemb": _to_blob(answer_embedding),
                    "qual": quality,
                    "quality_source": quality_source,
                    "judge_sampled": 1 if judge_sampled else 0,
                    "predicted_error_prob": predicted_error_prob,
                    "confidence_score": confidence_score,
                    "confidence_band": confidence_band,
                    "abstained": 1 if abstained else 0,
                    "abstain_reason": abstain_reason,
                    "grounded": 1 if grounded else 0,
                    "verification_status": verification_status,
                    "knowledge_version": knowledge_version,
                    "review_status": review_status,
                    "lat": latency_s,
                    "estimated_cost_usd": estimated_cost_usd,
                    "cost": estimated_cost_usd,
                    "rew": reward,
                    "decision_json": _safe_json(decision) if decision else None,
                    "correlation_id": correlation_id,
                    "quality_semantics": quality_semantics or _current_semantics(),
                    "q_tech": q_tech,
                    "q_calibrado": q_calibrado,
                    "p_entrega": p_entrega,
                    "detected_complexity": detected_complexity,
                    "ctx": context_label,
                    "tenant_id": tenant_id,
                    "payload": _safe_json(raw_payload),
                    "participant": participant,
                    "episode_id": episode_id,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "finish_reason": finish_reason,
                    "trace_json": _safe_json(trace) if trace else None,
                }.items()},
            )

        logger.info(
            "[query_service] log inserido: model=%s, modality=%s, reward=%s, quality=%s, estimated_cost_usd=%.4f",
            model, modality, reward, quality, estimated_cost_usd,
        )

    except SQLAlchemyError as exc:
        # Propaga: quem chama decide (o worker de feedback falha a tarefa, que fica visível), em vez de a linha
        # sumir com a tarefa em SUCCESS.
        logger.error(f"[query_service] erro ao inserir query_log: {exc}")
        raise
