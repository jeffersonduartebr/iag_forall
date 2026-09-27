# -*- coding: utf-8 -*-
# Objective: Idempotent seed of the reference data (model_pricing); the schema itself belongs to alembic_pg.
"""
db_manager.py (PRICING SEED)
------------------------------------------------------------------------------
O esquema do banco tem um único dono: a cadeia Alembic ``alembic_pg`` (``alembic -c alembic_pg.ini upgrade
head``). Este módulo não cria nem altera tabelas; ele só popula ``model_pricing`` com os preços de referência,
de forma idempotente (``ON CONFLICT (model) DO UPDATE``). Roda como ``python -m app.db_manager`` no ``db_init``
e como ``python /app/app/db_manager.py`` no prestart da imagem.
"""

from __future__ import annotations

import logging
import os
import sys

from sqlalchemy import text

if __package__ in (None, ""):  # executado como script: /app/app/db_manager.py
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reusa o engine singleton de db.py em vez de abrir um segundo pool de conexões (perf #25).
from app.db import engine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] db_manager: %(message)s",
)
logger = logging.getLogger("db_manager")

#: (modelo, US$ por 1k tokens de entrada, US$ por 1k tokens de saída)
PRICES = [
    # --- OpenAI ---
    ("gpt-5.1", 0.00125, 0.0100),
    ("gpt-5", 0.00125, 0.0100),
    ("gpt-5-mini", 0.00025, 0.0020),
    ("gpt-5-nano", 0.00005, 0.0004),
    ("gpt-4.1", 0.00200, 0.0080),
    ("gpt-4.1-mini", 0.00040, 0.0016),
    ("gpt-4o", 0.00250, 0.0100),
    ("gpt-4o-mini", 0.00015, 0.0006),
    # --- Google Gemini ---
    ("gemini-2.5-pro", 0.00125, 0.0100),
    ("gemini-2.5-flash", 0.00030, 0.0025),
    ("gemini-3.8-flash", 0.00075, 0.00375),  # Vertex AI, introdutório até 2026-12-31
    ("gemini-3.1-pro-preview", 0.00200, 0.0120),  # Vertex AI (global), juiz da execução em sombra
    # --- Anthropic Claude ---
    ("claude-opus-4.5", 0.00500, 0.0250),
    ("claude-sonnet-4.5", 0.00300, 0.0150),
    ("claude-haiku-4.5", 0.00100, 0.0050),
    # --- Local (custo elétrico estimado) ---
    ("phi4", 0.0000001, 0.0000001),
    ("mistral", 0.0000001, 0.0000001),
    ("gemma3", 0.0000001, 0.0000001),
    ("llama3.2-vision", 0.0000002, 0.0000002),
    ("llava", 0.0000002, 0.0000002),
    ("moondream", 0.0000001, 0.0000001),
]

#: Prefixos de namespace gravados para cada modelo, para o lookup casar com ou sem provedor.
NAMESPACES = ("", "openai/", "google/", "anthropic/", "ollama/")

SEED_SQL = text(
    """
    INSERT INTO model_pricing (model, cost_input_1k, cost_output_1k)
    VALUES (:m, :ci, :co)
    ON CONFLICT (model) DO UPDATE
    SET cost_input_1k = EXCLUDED.cost_input_1k, cost_output_1k = EXCLUDED.cost_output_1k
    """
)


def pricing_rows() -> list[dict]:
    """Every (namespace x model) row of the seed, as bind parameters."""
    return [
        {"m": f"{prefixo}{modelo}", "ci": entrada, "co": saida}
        for modelo, entrada, saida in PRICES
        for prefixo in NAMESPACES
    ]


def seed_pricing_data(conn) -> int:
    """Upsert the reference prices in one executemany; returns how many rows were sent."""
    linhas = pricing_rows()
    conn.execute(SEED_SQL, linhas)
    return len(linhas)


def initialize_system() -> None:
    """Seed only: the tables must already exist (``alembic -c alembic_pg.ini upgrade head``)."""
    logger.info("Atualizando a tabela de preços (model_pricing)...")
    with engine.begin() as conn:
        total = seed_pricing_data(conn)
    logger.info("model_pricing: %d linhas garantidas", total)


if __name__ == "__main__":
    initialize_system()
