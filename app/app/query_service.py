# -*- coding: utf-8 -*-
"""
query_service.py
----------------------------------------------------
Serviço de persistência para logs de consultas do roteador LLM.

Responsável por:
- Garantir a existência da tabela `query_log`
- Inserir registros de cada consulta avaliada (modelo, latência, custo, recompensa)
"""

import os
import logging
from sqlalchemy import create_engine, text

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s query_service: %(message)s",
)
logger = logging.getLogger("query_service")

# ============================================================
# Conexão com o banco
# ============================================================
DB_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:rootpass@mariadb:3306/routerdb"
)
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

# ============================================================
# Garantia da tabela query_log
# ============================================================
def ensure_query_log():
    """Cria a tabela query_log se não existir."""
    ddl = """
    CREATE TABLE IF NOT EXISTS query_log (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        query_text TEXT,
        chosen_model VARCHAR(255),
        answer MEDIUMTEXT,
        quality FLOAT,
        latency_s FLOAT,
        cost_per_1k FLOAT,
        reward FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
    except Exception as e:
        logger.warning(f"[query_service] Falha ao criar tabela query_log: {e}")

# ============================================================
# Inserção de log de consulta
# ============================================================
def insert_query_log(query, model, response, latency, cost, quality, reward):
    """
    Insere uma nova linha na tabela query_log.

    Args:
        query (str): Texto da consulta original.
        model (str): Nome do modelo escolhido.
        response (str): Resposta gerada.
        latency (float): Tempo de resposta em segundos.
        cost (float): Custo estimado por 1k tokens.
        quality (float): Nota de qualidade atribuída.
        reward (float): Recompensa calculada pelo Bandit.
    """
    ensure_query_log()
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO query_log
                    (query_text, chosen_model, answer,
                     latency_s, cost_per_1k, quality, reward)
                    VALUES (:q, :m, :a, :l, :c, :qual, :r)
                """),
                {
                    "q": query,
                    "m": model,
                    "a": response,
                    "l": latency,
                    "c": cost,
                    "qual": quality,
                    "r": reward,
                },
            )
        logger.info(f"[query_service] query_log inserido ({model}, reward={reward:.3f})")
    except Exception as e:
        logger.warning(f"[query_service] Falha ao inserir query_log: {e}")
