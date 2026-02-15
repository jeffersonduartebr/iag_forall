# -*- coding: utf-8 -*-
"""
query_service.py — versão MULTIMODAL COMPLETA
----------------------------------------------------
Serviço de persistência para logs avançados do Router Multimodal.

Agora inclui:
✔ modality (text, vision, multimodal)
✔ image_provided
✔ image_output_b64
✔ query_embedding (LONGBLOB)
✔ answer_embedding (LONGBLOB)
✔ raw_payload (LONGTEXT)
✔ context_label
✔ total compatibilidade com RAG multimodal

Compatível com:
- embeddings.py multimodal
- providers multimodais
- router_core multimodal
- judges multimodais
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
    """Get database engine from centralized module."""
    return get_engine()


# Lazy engine accessor for backward compatibility
engine = property(lambda self: _get_engine())


class _EngineProxy:
    """Proxy to centralized engine for backward compatibility."""

    def begin(self):
        """Resumo do comportamento desta função.

        Returns:
            Valor retornado pela função.
        """
        return get_engine().begin()

    def connect(self):
        """Resumo do comportamento desta função.

        Returns:
            Valor retornado pela função.
        """
        return get_engine().connect()

    def execute(self, *args, **kwargs):
        """Resumo do comportamento desta função.

        Args:
            *args: Parâmetro de entrada.
            **kwargs: Parâmetro de entrada.

        Returns:
            Valor retornado pela função.
        """
        return get_engine().execute(*args, **kwargs)


engine = _EngineProxy()


# ============================================================
# Helpers
# ============================================================

def _to_blob(vec) -> Optional[bytes]:
    """Converte listas/np arrays para bytes binários (LONGBLOB)."""
    if vec is None:
        return None
    try:
        return np.asarray(vec, dtype=np.float32).tobytes()
    except Exception:
        return None


def _safe_json(obj: dict | list | str | None) -> str:
    """Serializa payload multimodal em JSON seguro."""
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
        """Resumo do comportamento desta função.

        Args:
            value: Parâmetro de entrada.

        Returns:
            Valor retornado pela função.
        """
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
    """
    Cria a tabela query_log multimodal + embeddings + payload.
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
        latency_s FLOAT,
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
    cost_per_1k: float,
    quality: float,
    reward: float,
    context_label: Optional[str],
    raw_payload: dict | list | str | None = None,

    # embeddings
    query_embedding: Optional[List[float]] = None,
    answer_embedding: Optional[List[float]] = None,
) -> None:

    """Resumo do comportamento desta função.

    Args:
        query_text: Parâmetro de entrada.
        model: Parâmetro de entrada.
        modality: Parâmetro de entrada.
        image_provided: Parâmetro de entrada.
        answer: Parâmetro de entrada.
        image_output_b64: Parâmetro de entrada.
        latency_s: Parâmetro de entrada.
        cost_per_1k: Parâmetro de entrada.
        quality: Parâmetro de entrada.
        reward: Parâmetro de entrada.
        context_label: Parâmetro de entrada.
        raw_payload: Parâmetro de entrada.
        query_embedding: Parâmetro de entrada.
        answer_embedding: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
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
                     quality, latency_s, cost_per_1k, reward,
                     context_label, raw_payload)
                    VALUES
                    (:q, :m, :mod, :ip,
                     :ans, :img,
                     :qemb, :aemb,
                     :qual, :lat, :cost, :rew,
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
                    "lat": latency_s,
                    "cost": cost_per_1k,
                    "rew": reward,
                    "ctx": context_label,
                    "payload": _safe_json(raw_payload),
                }
            )

        logger.info(
            f"[query_service] log inserido: model={model}, modality={modality}, "
            f"reward={reward:.2f}, quality={quality:.2f}"
        )

    except SQLAlchemyError as exc:
        logger.warning(f"[query_service] erro ao inserir query_log: {exc}")
