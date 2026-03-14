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
from typing import Optional, List

import numpy as np
from sqlalchemy import create_engine, text
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
engine = property(lambda self: _get_engine())


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
    """Convert an embedding-like vector into the binary format stored in MySQL."""
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
                if str(k).lower() in sensitive_keys:
                    out[k] = "***REDACTED***"
                else:
                    out[k] = _redact(v)
            return out
        if isinstance(value, list):
            return [_redact(v) for v in value]
        return value

    try:
        return json.dumps(_redact(obj), ensure_ascii=False)
    except Exception:
        return "{}"


# ============================================================
# DDL — tabela multimodal EXTENDIDA
# ============================================================

def ensure_query_log() -> None:
    """Create the `query_log` table when it does not already exist.

    The table definition supports text-only and multimodal requests, optional
    image outputs, serialized payloads, and binary embeddings used by later
    analytics or judging flows.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS query_log (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,

        -- texto original
        query_text TEXT,

        -- modelo escolhido
        chosen_model VARCHAR(255) NOT NULL,

        -- multimodalidade
        modality VARCHAR(32) DEFAULT 'text',
        image_provided TINYINT DEFAULT 0,

        -- outputs
        answer LONGTEXT,
        image_output_b64 LONGTEXT,

        -- embeddings
        query_embedding LONGBLOB,
        answer_embedding LONGBLOB,

        -- metadados
        quality FLOAT,
        quality_source VARCHAR(32) DEFAULT 'unknown',
        judge_sampled TINYINT DEFAULT 0,
        predicted_error_prob FLOAT NULL,
        confidence_score FLOAT NULL,
        confidence_band VARCHAR(16) DEFAULT NULL,
        abstained TINYINT DEFAULT 0,
        abstain_reason VARCHAR(64) DEFAULT NULL,
        grounded TINYINT DEFAULT 0,
        verification_status VARCHAR(32) DEFAULT NULL,
        knowledge_version VARCHAR(255) DEFAULT NULL,
        review_status VARCHAR(32) DEFAULT NULL,
        latency_s FLOAT,
        estimated_cost_usd FLOAT NULL,
        cost_per_1k FLOAT,
        reward FLOAT,

        -- contexto semântico opcional
        context_label VARCHAR(64),

        -- payload multimodal COMPLETO
        raw_payload LONGTEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_created_at (created_at),
        INDEX idx_model (chosen_model),
        INDEX idx_modality (modality)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
            # Keep existing deployments compatible when the table predates the
            # newer feedback attribution fields.
            conn.execute(
                text(
                    """
                    ALTER TABLE query_log
                    ADD COLUMN IF NOT EXISTS quality_source VARCHAR(32) DEFAULT 'unknown'
                    """
                )
            )
            conn.execute(
                text(
                    """
                    ALTER TABLE query_log
                    ADD COLUMN IF NOT EXISTS judge_sampled TINYINT DEFAULT 0
                    """
                )
            )
            conn.execute(
                text(
                    """
                    ALTER TABLE query_log
                    ADD COLUMN IF NOT EXISTS predicted_error_prob FLOAT NULL
                    """
                )
            )
            conn.execute(text("ALTER TABLE query_log ADD COLUMN IF NOT EXISTS confidence_score FLOAT NULL"))
            conn.execute(text("ALTER TABLE query_log ADD COLUMN IF NOT EXISTS confidence_band VARCHAR(16) DEFAULT NULL"))
            conn.execute(text("ALTER TABLE query_log ADD COLUMN IF NOT EXISTS abstained TINYINT DEFAULT 0"))
            conn.execute(text("ALTER TABLE query_log ADD COLUMN IF NOT EXISTS abstain_reason VARCHAR(64) DEFAULT NULL"))
            conn.execute(text("ALTER TABLE query_log ADD COLUMN IF NOT EXISTS grounded TINYINT DEFAULT 0"))
            conn.execute(text("ALTER TABLE query_log ADD COLUMN IF NOT EXISTS verification_status VARCHAR(32) DEFAULT NULL"))
            conn.execute(text("ALTER TABLE query_log ADD COLUMN IF NOT EXISTS knowledge_version VARCHAR(255) DEFAULT NULL"))
            conn.execute(text("ALTER TABLE query_log ADD COLUMN IF NOT EXISTS review_status VARCHAR(32) DEFAULT NULL"))
            conn.execute(text("ALTER TABLE query_log ADD COLUMN IF NOT EXISTS estimated_cost_usd FLOAT NULL"))
        logger.info("[query_service] Tabela 'query_log' pronta (EXTENDIDA multimodal).")
    except SQLAlchemyError as exc:
        logger.warning("[query_service] Falha ao criar tabela query_log: %s", exc)


# ============================================================
# Inserção multimodal completa
# ============================================================

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
    quality: float,
    reward: float,
    quality_source: str = "unknown",
    judge_sampled: bool = False,
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
    raw_payload: dict | list | str | None = None,

    # embeddings
    query_embedding: Optional[List[float]] = None,
    answer_embedding: Optional[List[float]] = None,
) -> None:
    """Insert one fully-populated router execution record into `query_log`.

    Callers provide the normalized execution summary plus any optional
    multimodal payloads and embeddings. The function ensures the table exists,
    redacts sensitive payload fields, and stores binary vectors in the compact
    representation expected by the schema.
    """
    ensure_query_log()

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
                     context_label, raw_payload)
                    VALUES
                     (:q, :m, :mod, :ip,
                     :ans, :img,
                     :qemb, :aemb,
                     :qual, :quality_source, :judge_sampled, :predicted_error_prob,
                     :confidence_score, :confidence_band, :abstained, :abstain_reason,
                     :grounded, :verification_status, :knowledge_version, :review_status,
                     :lat, :estimated_cost_usd, :cost, :rew,
                     :ctx, :payload)
                """),
                {
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
                    "ctx": context_label,
                    "payload": _safe_json(raw_payload),
                }
            )

        logger.info(
            f"[query_service] log inserido: model={model}, modality={modality}, "
            f"reward={reward:.2f}, quality={quality:.2f}, estimated_cost_usd={estimated_cost_usd:.4f}"
        )

    except SQLAlchemyError as exc:
        logger.warning(f"[query_service] erro ao inserir query_log: {exc}")
