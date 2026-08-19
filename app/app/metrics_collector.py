# Objective: Application runtime code for metrics collector.
# app/metrics_collector.py
# -*- coding: utf-8 -*-
"""
Coletor de métricas multimodais para o Router LLM.
--------------------------------------------------
Esta versão:
  ✓ Registra métricas separadas por modalidade (text/vision/multimodal)
  ✓ Mantém EMA leve em memória (por modelo + modalidade)
  ✓ Persiste métricas em MariaDB (tabela model_metrics)
  ✓ Suporta custo_per_1k tokens e custo por consulta
  ✓ Guarda tokens_in, tokens_out, embedding_dim, vision_usage
"""

import threading
from typing import Dict, Optional

from sqlalchemy import text

from app.db import get_engine

# Thread-safety para EMA in-memory
_LOCK = threading.Lock()

# Estrutura:
# _MODEL_METRICS[(model, modality)] = {
#     "quality": float,
#     "latency": float,
#     "cost": float,
# }
_MODEL_METRICS: Dict[tuple, Dict[str, float]] = {}

_table_ready = False


def _engine():
    """Return the shared application database engine."""
    return get_engine()


# ============================================================
# TABLE CREATION
# ============================================================

def _ensure_model_metrics_table() -> None:
    """Create model_metrics table once at startup."""
    global _table_ready
    if _table_ready:
        return
    ddl = """
    CREATE TABLE IF NOT EXISTS model_metrics (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,

      model VARCHAR(255) NOT NULL,
      modality VARCHAR(32) NOT NULL DEFAULT 'text',

      latency_ms FLOAT NOT NULL,
      cost_usd FLOAT NOT NULL,
      cost_per_1k FLOAT DEFAULT 0,
      quality_score FLOAT NOT NULL,

      tokens_in INT DEFAULT NULL,
      tokens_out INT DEFAULT NULL,

      embedding_dim INT DEFAULT NULL,
      vision_usage TINYINT DEFAULT 0,

      fitness FLOAT NOT NULL,
      generation INT NOT NULL DEFAULT 0,
      timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

      INDEX idx_model_modality (model, modality, timestamp),
      INDEX idx_ts (timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    with _engine().begin() as conn:
        conn.execute(text(ddl))
    _table_ready = True


# ============================================================
# PERSISTÊNCIA DE LINHA COMPLETA
# ============================================================

def _persist_sample(
    model_name: str,
    modality: str,
    latency_s: float,
    quality_0_10: float,
    cost_usd: float,
    cost_per_1k: float = 0.0,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    vision_usage: bool = False,
    generation: int = 0,
) -> None:
    """
    Persiste uma linha de métricas multimodais no MariaDB.
    """
    try:
        _ensure_model_metrics_table()

        latency_ms = float(latency_s) * 1000.0
        quality = float(quality_0_10)

        # Fitness multimodal: pesada em qualidade + leve em latência
        speed_term = 1.0 - min(latency_s, 12.0) / 12.0
        fitness = max(0.0, min(1.0,
            (quality / 10.0) * 0.75 + speed_term * 0.25
        ))

        with _engine().begin() as conn:
            conn.execute(
                text("""
                INSERT INTO model_metrics
                (model, modality, latency_ms, cost_usd, cost_per_1k,
                 quality_score, tokens_in, tokens_out,
                 embedding_dim, vision_usage,
                 fitness, generation)
                VALUES
                (:model, :modality, :latms, :cost, :cost1k,
                 :qual, :tin, :tout, :embdim, :vis,
                 :fit, :gen)
                """),
                {
                    "model": model_name,
                    "modality": modality,
                    "latms": latency_ms,
                    "cost": cost_usd,
                    "cost1k": cost_per_1k,
                    "qual": quality,
                    "tin": tokens_in,
                    "tout": tokens_out,
                    "embdim": embedding_dim,
                    "vis": 1 if vision_usage else 0,
                    "fit": fitness,
                    "gen": generation,
                }
            )

    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("[metrics_collector] persist failed: %s", exc)


# ============================================================
# EMA MULTIMODAL (memória)
# ============================================================

def update_model_metrics(
    model_name: str,
    latency: float,
    quality: float,
    cost: float,
    *,
    modality: str = "text",
    cost_per_1k: float = 0.0,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    vision_usage: bool = False,
    generation: int = 0,
) -> None:
    """
    Atualiza snapshot e persiste uma amostra multimodal.
    """
    key = (model_name, modality)

    # EMA leve (thread-safe)
    with _LOCK:
        if key not in _MODEL_METRICS:
            _MODEL_METRICS[key] = {
                "quality": round(float(quality), 3),
                "latency": round(float(latency), 3),
                "cost": round(float(cost), 6),
            }
        else:
            prev = _MODEL_METRICS[key]
            _MODEL_METRICS[key] = {
                "quality": round(prev["quality"] * 0.7 + float(quality) * 0.3, 3),
                "latency": round(prev["latency"] * 0.7 + float(latency) * 0.3, 3),
                "cost": round(prev["cost"] * 0.7 + float(cost) * 0.3, 6),
            }

    # Escrita assíncrona no banco
    _persist_sample(
        model_name=model_name,
        modality=modality,
        latency_s=float(latency),
        quality_0_10=float(quality),
        cost_usd=float(cost),
        cost_per_1k=float(cost_per_1k),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        embedding_dim=embedding_dim,
        vision_usage=vision_usage,
        generation=generation,
    )


# ============================================================
# SNAPSHOT API (para router_strategy)
# ============================================================

def get_snapshot() -> Dict[tuple, Dict[str, float]]:
    """Return snapshot.

This helper centralizes retrieval logic so callers do not have to duplicate lookup behavior."""
    with _LOCK:
        return {k: v.copy() for k, v in _MODEL_METRICS.items()}
