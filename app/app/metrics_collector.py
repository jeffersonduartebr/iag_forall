# app/metrics_collector.py
# -*- coding: utf-8 -*-
"""
Armazena métricas dinâmicas por modelo em memória (EMA leve) e
persiste amostras no MariaDB em tabela 'model_metrics', compatível
com o leitor de correlação.
"""

import os
import threading
from typing import Dict

from sqlalchemy import create_engine, text

_LOCK = threading.Lock()
_MODEL_METRICS: Dict[str, Dict[str, float]] = {}

DB_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:rootpass@mariadb:3306/routerdb",
)
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)


def _ensure_model_metrics_table() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS model_metrics (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      model VARCHAR(255) NOT NULL,
      latency_ms FLOAT NOT NULL,
      cost_usd FLOAT NOT NULL,
      quality_score FLOAT NOT NULL,
      fitness FLOAT NOT NULL,
      generation INT NOT NULL DEFAULT 0,
      timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      INDEX idx_ts (timestamp),
      INDEX idx_model_ts (model, timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _persist_sample(
    model_name: str,
    latency_s: float,
    quality_0_10: float,
    cost_usd_per_query: float,
    generation: int = 0,
) -> None:
    """
    Persiste uma linha na model_metrics.
    Converte latência para ms e normaliza 'fitness' de forma simples.
    """
    try:
        _ensure_model_metrics_table()
        latency_ms = float(latency_s) * 1000.0
        quality = float(quality_0_10)

        # Fitness simples: 70% qualidade (0..10 → 0..1), 30% rapidez (<=10s)
        speed_term = 1.0 - min(float(latency_s), 10.0) / 10.0
        fitness = max(0.0, min(1.0, (quality / 10.0) * 0.7 + speed_term * 0.3))

        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO model_metrics
                    (model, latency_ms, cost_usd, quality_score, fitness, generation)
                    VALUES (:m, :latms, :cost, :qual, :fit, :gen)
                    """
                ),
                {
                    "m": model_name,
                    "latms": latency_ms,
                    "cost": float(cost_usd_per_query),
                    "qual": quality,
                    "fit": fitness,
                    "gen": int(generation),
                },
            )
    except Exception as exc:  # pragma: no cover
        # Mantém a app resiliente mesmo se o banco cair momentaneamente
        # (podemos logar no logger global se preferir)
        pass


def update_model_metrics(model_name: str, latency: float, quality: float, cost: float):
    """
    Atualiza as métricas dinâmicas de um modelo (EMA leve) e
    persiste uma amostra na tabela 'model_metrics'.
    - latency: segundos
    - quality: 0..10 (compatível com seu pipeline atual)
    - cost: USD por consulta (não por mil tokens)
    """
    # Atualização do snapshot in-memory (EMA leve)
    with _LOCK:
        if model_name not in _MODEL_METRICS:
            _MODEL_METRICS[model_name] = {
                "quality": round(float(quality), 3),
                "latency": round(float(latency), 3),
                "cost": round(float(cost), 6),
            }
        else:
            prev = _MODEL_METRICS[model_name]
            _MODEL_METRICS[model_name] = {
                "quality": round(prev["quality"] * 0.7 + float(quality) * 0.3, 3),
                "latency": round(prev["latency"] * 0.7 + float(latency) * 0.3, 3),
                "cost": round(prev["cost"] * 0.7 + float(cost) * 0.3, 6),
            }

    # Persistência de amostra para analytics/correlação
    _persist_sample(
        model_name=model_name,
        latency_s=float(latency),
        quality_0_10=float(quality),
        cost_usd_per_query=float(cost),
        generation=0,
    )


def get_snapshot() -> Dict[str, Dict[str, float]]:
    """Retorna uma cópia das métricas atuais (thread-safe)."""
    with _LOCK:
        return {k: v.copy() for k, v in _MODEL_METRICS.items()}
