# -*- coding: utf-8 -*-
"""
db_manager.py
----------------------------------------------------
Inicializa e migra o banco de dados do Router LLM.

Cria/atualiza automaticamente as tabelas para:
- EMA (ema_history / ema_history_log)
- Bandit (bandit_context_stats)
- Logs de consultas (query_log)
- Cache semântico (semantic_cache)
- Juízes (judge_logs / judge_performance_log)
- Métricas por modelo (model_metrics)
- NSGA (nsga_weights / nsga_fallback_log)

Também executa opcionalmente o script `db/init_db.sql`, se existir.
"""

from __future__ import annotations

import os
import logging
from typing import Dict

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] db_manager: %(message)s",
)
logger = logging.getLogger("db_manager")

# ---------------------------------------------------------------------
# Settings e engine
# ---------------------------------------------------------------------

try:
    # Execução como pacote (app/db_manager.py)
    from app.settings_dynamic import settings
except Exception:
    # Execução direta: ajusta sys.path e tenta novamente
    import sys

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )
    from app.settings_dynamic import settings  # type: ignore

DB_URL = (
    f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}"
    f"@{settings.DB_HOST}:3306/{settings.DB_NAME}"
)

engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

# ---------------------------------------------------------------------
# DDL (CREATE TABLE IF NOT EXISTS ...)
# Obs.: cada instrução é executada separadamente (MariaDB não aceita
# múltiplos CREATE no mesmo execute).
# ---------------------------------------------------------------------

TABLES_DDL: Dict[str, str] = {
    "ema_history": """
        CREATE TABLE IF NOT EXISTS ema_history (
            model VARCHAR(255) PRIMARY KEY,
            ema_latency FLOAT NOT NULL,
            ema_quality FLOAT NOT NULL,
            ema_cost FLOAT NOT NULL,
            updates INT DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci;
    """,
    "ema_history_log": """
        CREATE TABLE IF NOT EXISTS ema_history_log (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            model VARCHAR(255) NOT NULL,
            ema_latency FLOAT NOT NULL,
            ema_quality FLOAT NOT NULL,
            ema_cost FLOAT NOT NULL,
            update_num INT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_model_created_at (model, created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci;
    """,
    "bandit_context_stats": """
        CREATE TABLE IF NOT EXISTS bandit_context_stats (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            context_type VARCHAR(50) NOT NULL,
            model VARCHAR(255) NOT NULL,
            avg_reward FLOAT DEFAULT 0,
            count INT DEFAULT 0,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uq_ctx_model (context_type, model)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci;
    """,
    "query_log": """
        CREATE TABLE IF NOT EXISTS query_log (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            query_text TEXT NOT NULL,
            chosen_model VARCHAR(255) NOT NULL,
            answer MEDIUMTEXT,
            quality FLOAT,
            latency_s FLOAT,
            cost_per_1k FLOAT,
            reward FLOAT,
            context_label VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci;
    """,
    "semantic_cache": """
        CREATE TABLE IF NOT EXISTS semantic_cache (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            query_hash VARCHAR(64) UNIQUE,
            query_text TEXT,
            context_category VARCHAR(50),
            answer MEDIUMTEXT,
            model_used VARCHAR(255),
            embedding LONGBLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci;
    """,
    "judge_logs": """
        CREATE TABLE IF NOT EXISTS judge_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            query TEXT,
            answer TEXT,
            judge_model VARCHAR(255),
            score_before FLOAT,
            fallback_model VARCHAR(255),
            score_after FLOAT,
            event_type VARCHAR(50)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci;
    """,
    "judge_performance_log": """
        CREATE TABLE IF NOT EXISTS judge_performance_log (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            judge_model VARCHAR(255) NOT NULL,
            avg_score FLOAT DEFAULT 0,
            avg_latency FLOAT DEFAULT 0,
            avg_cost FLOAT DEFAULT 0,
            consistency FLOAT DEFAULT 0,
            fitness FLOAT DEFAULT 0,
            window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            window_end TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_judge_model (judge_model),
            INDEX idx_window_end (window_end)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci;
    """,
    "model_metrics": """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            model VARCHAR(255) NOT NULL,
            latency_ms FLOAT DEFAULT 0,
            cost_usd FLOAT DEFAULT 0,
            quality_score FLOAT DEFAULT 0,
            fitness FLOAT DEFAULT 0,
            generation INT DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_model_timestamp (model, timestamp)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci;
    """,
    "nsga_weights": """
        CREATE TABLE IF NOT EXISTS nsga_weights (
            model VARCHAR(255) PRIMARY KEY,
            weight FLOAT NOT NULL,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci;
    """,
    "nsga_fallback_log": """
        CREATE TABLE IF NOT EXISTS nsga_fallback_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            old_model VARCHAR(255),
            new_model VARCHAR(255),
            event_type VARCHAR(64),
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci;
    """,
}

# ---------------------------------------------------------------------
# Execução opcional de script SQL externo
# ---------------------------------------------------------------------


def _execute_init_sql() -> None:
    """
    Executa o script `db/init_db.sql`, se existir.
    Divide o conteúdo por ';' e executa cada instrução.
    """
    sql_path = os.path.join(os.getcwd(), "db", "init_db.sql")
    if not os.path.exists(sql_path):
        logger.info("[db_manager] Nenhum init_db.sql encontrado.")
        return

    try:
        with open(sql_path, "r", encoding="utf-8") as file:
            sql_script = file.read()

        with engine.begin() as conn:
            for stmt in sql_script.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))

        logger.info("[db_manager] init_db.sql executado com sucesso.")
    except Exception as exc:
        logger.error("[db_manager] Erro ao executar init_db.sql: %s", exc)

# ---------------------------------------------------------------------
# Criação das tabelas
# ---------------------------------------------------------------------


def initialize_tables() -> None:
    """Cria/verifica todas as tabelas necessárias."""
    logger.info("[db_manager] Verificando/criando tabelas...")
    try:
        with engine.begin() as conn:
            for name, ddl in TABLES_DDL.items():
                try:
                    conn.execute(text(ddl))
                    logger.info("✅ Tabela verificada/criada: %s", name)
                except SQLAlchemyError as exc:
                    logger.error(
                        "[db_manager] Falha ao criar tabela %s: %s",
                        name,
                        exc,
                    )
    except SQLAlchemyError as exc:
        logger.error("[db_manager] Erro geral ao criar tabelas: %s", exc)

# ---------------------------------------------------------------------
# Inicialização completa
# ---------------------------------------------------------------------


def initialize_system() -> None:
    """Executa a inicialização completa do banco."""
    logger.info("🚀 Iniciando setup do banco de dados...")
    initialize_tables()
    _execute_init_sql()
    logger.info("✅ Banco de dados inicializado com sucesso.")

# ---------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------

if __name__ == "__main__":
    initialize_system()
    logger.info("[db_manager] Inicialização concluída com sucesso.")
