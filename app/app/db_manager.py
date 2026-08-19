# -*- coding: utf-8 -*-
# Objective: Application runtime code for db manager.
"""
db_manager.py (FULL SCHEMA + PRICING SEED)
------------------------------------------------------------------------------
Gerenciador mestre do Banco de Dados.
Responsável por:
1. Criar todas as tabelas necessárias.
2. Verificar e criar colunas (Schema Evolution).
3. Popular dados iniciais de precificação (Seed).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# Reusa o engine singleton de db.py em vez de abrir um segundo pool de conexões,
# evitando fragmentar o orçamento de conexões do MariaDB (perf #25).
from app.db import engine

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] db_manager: %(message)s",
)
logger = logging.getLogger("db_manager")

# ---------------------------------------------------------------------
# 1. Conexão com Banco (Lê de settings_dynamic ou ENV)
# ---------------------------------------------------------------------
try:
    from app.settings_dynamic import settings
except Exception:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from app.settings_dynamic import settings

DB_HOST = settings.DB_HOST
DB_PORT = settings.DB_PORT
DB_USER = settings.DB_USER
DB_PASS = settings.DB_PASS
DB_NAME = settings.DB_NAME

# URL de conexão SQLAlchemy (mantida para diagnóstico/logs)
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

logger.info(f"[db_manager] Conectando em {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}...")


# ---------------------------------------------------------------------
# 2. Definição do Schema (Tabelas e suas Colunas Críticas)
# ---------------------------------------------------------------------

SCHEMA_DEFINITIONS: dict[str, dict[str, Any]] = {
    # ============================================================
    # 🧠 1. CACHE SEMÂNTICO
    # ============================================================
    "semantic_cache": {
        "ddl": """
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                query_hash VARCHAR(255) NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        "columns": {
            "modality": "VARCHAR(32) DEFAULT 'text'",
            "query_text": "TEXT",
            "answer": "LONGTEXT",
            "model_used": "VARCHAR(255)",
            "embedding": "LONGBLOB",
            "image_output_b64": "LONGTEXT"
        }
    },

    # ============================================================
    # 📝 2. QUERY LOG (Histórico Completo)
    # ============================================================
    "query_log": {
        "ddl": """
            CREATE TABLE IF NOT EXISTS query_log (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        "columns": {
            "query_text": "TEXT",
            "chosen_model": "VARCHAR(255) NOT NULL DEFAULT 'unknown'",
            "modality": "VARCHAR(32) DEFAULT 'text'",
            "image_provided": "TINYINT DEFAULT 0",
            "answer": "LONGTEXT",
            "image_output_b64": "LONGTEXT",
            "query_embedding": "LONGBLOB",
            "answer_embedding": "LONGBLOB",
            "quality": "FLOAT",
            "latency_s": "FLOAT",
            "cost_per_1k": "FLOAT",
            "reward": "FLOAT",
            "context_label": "VARCHAR(64)",
            "raw_payload": "LONGTEXT"
        }
    },

    # ============================================================
    # 📈 3. EMA HISTORY (Métricas de Aprendizado)
    # ============================================================
    "ema_history": {
        "ddl": """
            CREATE TABLE IF NOT EXISTS ema_history (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                model VARCHAR(255) NOT NULL,
                modality ENUM('text','vision','multimodal') NOT NULL DEFAULT 'text',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_model_modality (model, modality)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        "columns": {
            "ema_latency": "FLOAT NOT NULL DEFAULT 0",
            "ema_cost": "FLOAT NOT NULL DEFAULT 0",
            "ema_quality": "FLOAT NOT NULL DEFAULT 0",
            "ema_alignment": "FLOAT NOT NULL DEFAULT 0"
        }
    },

    "ema_history_log": {
        "ddl": """
            CREATE TABLE IF NOT EXISTS ema_history_log (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_created (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        "columns": {
            "modality": "ENUM('text','vision','multimodal') NOT NULL DEFAULT 'text'",
            "model": "VARCHAR(255) NOT NULL",
            "ema_latency": "FLOAT NOT NULL",
            "ema_cost": "FLOAT NOT NULL",
            "ema_quality": "FLOAT NOT NULL",
            "ema_alignment": "FLOAT NOT NULL",
            "update_num": "INT NOT NULL DEFAULT 0"
        }
    },

    # ============================================================
    # 📊 4. MODEL METRICS (Raw Data para NSGA)
    # ============================================================
    "model_metrics": {
        "ddl": """
            CREATE TABLE IF NOT EXISTS model_metrics (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_ts (timestamp)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        "columns": {
            "model": "VARCHAR(255) NOT NULL",
            "modality": "VARCHAR(32) NOT NULL DEFAULT 'text'",
            "latency_ms": "FLOAT NOT NULL DEFAULT 0",
            "cost_usd": "FLOAT NOT NULL DEFAULT 0",
            "cost_per_1k": "FLOAT DEFAULT 0",
            "quality_score": "FLOAT NOT NULL DEFAULT 0",
            "tokens_in": "INT DEFAULT NULL",
            "tokens_out": "INT DEFAULT NULL",
            "embedding_dim": "INT DEFAULT NULL",
            "vision_usage": "TINYINT DEFAULT 0",
            "fitness": "FLOAT NOT NULL DEFAULT 0",
            "generation": "INT NOT NULL DEFAULT 0"
        }
    },

    # ============================================================
    # ⚖️ 5. JUDGES (Auditoria)
    # ============================================================
    "judge_logs": {
        "ddl": """
            CREATE TABLE IF NOT EXISTS judge_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        "columns": {
            "query": "TEXT",
            "answer": "TEXT",
            "judge_model": "VARCHAR(255)",
            "score_before": "FLOAT",
            "fallback_model": "VARCHAR(255)",
            "score_after": "FLOAT",
            "event_type": "VARCHAR(50)",
            "modality": "VARCHAR(32) DEFAULT 'text'",
            "image_hash": "VARCHAR(128) NULL"
        }
    },

    "judge_performance_log": {
        "ddl": """
            CREATE TABLE IF NOT EXISTS judge_performance_log (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        "columns": {
            "judge_model": "VARCHAR(255) NOT NULL",
            "avg_score": "FLOAT DEFAULT 0",
            "avg_latency": "FLOAT DEFAULT 0",
            "avg_cost": "FLOAT DEFAULT 0",
            "consistency": "FLOAT DEFAULT 0",
            "fitness": "FLOAT DEFAULT 0",
            "window_start": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "window_end": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        }
    },

    # ============================================================
    # 🎰 6. BANDITS (Reinforcement Learning)
    # ============================================================
    "bandit_context_stats": {
        "ddl": """
            CREATE TABLE IF NOT EXISTS bandit_context_stats (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                context_label VARCHAR(255) NOT NULL,
                model VARCHAR(255) NOT NULL,
                last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uq_ctx_model (context_label, model)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        "columns": {
            "avg_reward": "FLOAT DEFAULT 0",
            "count": "INT DEFAULT 0",
            "var": "FLOAT DEFAULT 0",
            "M2": "FLOAT DEFAULT 0"
        }
    },

    # ============================================================
    # 🧬 7. NSGA-II (Pesos e Parâmetros)
    # ============================================================
    "nsga_weights": {
        "ddl": """
            CREATE TABLE IF NOT EXISTS nsga_weights (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                model VARCHAR(255) NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        "columns": {
            "modality": "VARCHAR(32) NOT NULL DEFAULT 'text'",
            "weight": "FLOAT NOT NULL DEFAULT 0"
        }
    },

    "nsga_params": {
        "ddl": """
            CREATE TABLE IF NOT EXISTS nsga_params (
                id INT PRIMARY KEY,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        "columns": {
            "modality": "VARCHAR(32) NOT NULL DEFAULT 'text'",
            "N_pop": "INT NOT NULL DEFAULT 40",
            "N_gen": "INT NOT NULL DEFAULT 20",
            "cxpb": "FLOAT NOT NULL DEFAULT 0.9",
            "mutpb": "FLOAT NOT NULL DEFAULT 0.1",
            "eta_c": "FLOAT NOT NULL DEFAULT 20.0",
            "eta_m": "FLOAT NOT NULL DEFAULT 20.0"
        }
    },

    "nsga_meta_results": {
        "ddl": """
            CREATE TABLE IF NOT EXISTS nsga_meta_results (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        "columns": {
            "trial_id": "INT NOT NULL",
            "modality": "VARCHAR(32) NOT NULL DEFAULT 'text'",
            "N_pop": "INT NOT NULL",
            "N_gen": "INT NOT NULL",
            "cxpb": "FLOAT NOT NULL",
            "mutpb": "FLOAT NOT NULL",
            "eta_c": "FLOAT NOT NULL",
            "eta_m": "FLOAT NOT NULL",
            "eff_mean": "FLOAT NOT NULL",
            "eff_std": "FLOAT NOT NULL"
        }
    },

    # ============================================================
    # ⚙️ 8. SETTINGS & REGISTRY
    # ============================================================
    "settings_dynamic": {
        "ddl": """
            CREATE TABLE IF NOT EXISTS settings_dynamic (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                setting_key VARCHAR(512) NOT NULL UNIQUE,
                setting_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        "columns": {}
    },

    "model_registry": {
        "ddl": """
            CREATE TABLE IF NOT EXISTS model_registry (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                model VARCHAR(255) NOT NULL,
                is_active TINYINT DEFAULT 1
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        "columns": {
            "modality": "ENUM('text','vision','multimodal') NOT NULL DEFAULT 'text'",
            "provider": "VARCHAR(64) NOT NULL DEFAULT 'ollama'",
            "max_context": "INT DEFAULT 4096",
            "cost_per_1k": "FLOAT DEFAULT 0"
        }
    },
    # ============================================================
    # 9. TABELA DE PREÇOS DINÂMICA
    # ============================================================
    "model_pricing": {
        "ddl": """
            CREATE TABLE IF NOT EXISTS model_pricing (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                model VARCHAR(255) NOT NULL UNIQUE,
                cost_input_1k FLOAT DEFAULT 0.0,
                cost_output_1k FLOAT DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        "columns": {
            "cost_input_1k": "FLOAT DEFAULT 0.0",
            "cost_output_1k": "FLOAT DEFAULT 0.0"
        }
    }
}


# ---------------------------------------------------------------------
# 3. Seed de Preços (Novos Modelos 2025)
# ---------------------------------------------------------------------
def seed_pricing_data(conn):
    """
    Popula a tabela model_pricing com valores de mercado atualizados.
    Converte preços de 1M tokens para 1k tokens.
    """
    # (Nome Modelo, Input $/1k, Output $/1k)
    PRICES = [
        # --- OpenAI (Novos modelos GPT-5 e GPT-4 atualizados) ---
        ("gpt-5.1", 0.00125, 0.0100),
        ("gpt-5", 0.00125, 0.0100),
        ("gpt-5-mini", 0.00025, 0.0020),
        ("gpt-5-nano", 0.00005, 0.0004),
        ("gpt-4.1", 0.00200, 0.0080),
        ("gpt-4.1-mini", 0.00040, 0.0016),
        ("gpt-4o", 0.00250, 0.0100),
        ("gpt-4o-mini", 0.00015, 0.0006),

        # --- Google Gemini (Novos modelos 2.5) ---
        ("gemini-2.5-pro", 0.00125, 0.0100),
        ("gemini-2.5-flash", 0.00030, 0.0025),

        # --- Anthropic Claude (Novos modelos 4.5) ---
        ("claude-opus-4.5", 0.00500, 0.0250),
        ("claude-sonnet-4.5", 0.00300, 0.0150),
        ("claude-haiku-4.5", 0.00100, 0.0050),

        # --- Local (Custo Elétrico Estimado - Baixo) ---
        ("phi4", 0.0000001, 0.0000001),
        ("mistral", 0.0000001, 0.0000001),
        ("gemma3", 0.0000001, 0.0000001),
        ("llama3.2-vision", 0.0000002, 0.0000002),
        ("llava", 0.0000002, 0.0000002),
        ("moondream", 0.0000001, 0.0000001),
    ]

    logger.info("💰 Atualizando tabela de preços (model_pricing)...")
    for model, inp, out in PRICES:
        # Adiciona variações de namespace para garantir match
        variations = [model, f"openai/{model}", f"google/{model}", f"anthropic/{model}", f"ollama/{model}"]
        for v in variations:
            try:
                conn.execute(
                    text("""
                        INSERT INTO model_pricing (model, cost_input_1k, cost_output_1k)
                        VALUES (:m, :ci, :co)
                        ON DUPLICATE KEY UPDATE cost_input_1k = :ci, cost_output_1k = :co
                    """),
                    {"m": v, "ci": inp, "co": out}
                )
            except SQLAlchemyError:
                pass


# ---------------------------------------------------------------------
# 4. Funções de Inicialização e Migração
# ---------------------------------------------------------------------

def ensure_table(table_name: str, ddl: str, conn):
    """Cria a tabela se não existir."""
    try:
        conn.execute(text(ddl))
    except SQLAlchemyError as e:
        logger.error(f"❌ Erro ao criar tabela '{table_name}': {e}")

def ensure_columns(table_name: str, columns: Dict[str, str], conn):
    """Verifica se as colunas existem. Se não, cria."""
    if not columns:
        return

    for col_name, col_def in columns.items():
        try:
            # MariaDB 10.2+ suporta ADD COLUMN IF NOT EXISTS
            alter_sql = f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {col_name} {col_def};"
            conn.execute(text(alter_sql))
        except SQLAlchemyError as e:
            logger.warning(f"⚠️ Erro ao verificar coluna '{col_name}' em '{table_name}': {e}")


def ensure_indexes(conn):
    """Create performance indexes (Quick Win #3)."""
    indexes = [
        # Semantic Cache: optimize lookup by modality + query_hash
        ("idx_semantic_cache_lookup", "semantic_cache", "(modality, query_hash)"),
        # EMA History Log: optimize cleanup queries
        ("idx_ema_log_created", "ema_history_log", "(created_at)"),
        # Query Log: optimize by model and created_at for analytics
        ("idx_query_log_model", "query_log", "(chosen_model, created_at)"),
    ]

    for idx_name, table, columns in indexes:
        try:
            # MariaDB/MySQL syntax for CREATE INDEX IF NOT EXISTS
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS {idx_name} ON {table} {columns}
            """))
            logger.info(f"✅ Index '{idx_name}' on '{table}' ensured")
        except SQLAlchemyError as e:
            # Index might already exist with different definition, ignore
            if "Duplicate" not in str(e):
                logger.warning(f"⚠️ Error creating index '{idx_name}': {e}")


def initialize_system() -> None:
    """Execute the initialize system routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    logger.info("🚀 Iniciando verificação e migração do Schema...")

    try:
        with engine.begin() as conn:
            for table_name, schema in SCHEMA_DEFINITIONS.items():
                # 1. Criar tabela base
                ensure_table(table_name, schema["ddl"], conn)

                # 2. Garantir todas as colunas (migração)
                ensure_columns(table_name, schema.get("columns", {}), conn)

            # 3. Popular Preços
            seed_pricing_data(conn)

            # 4. Criar índices de performance (Quick Win #3)
            ensure_indexes(conn)

        logger.info("✅ Schema do banco de dados e preços atualizados com sucesso!")

    except SQLAlchemyError as e:
        logger.critical(f"🔥 Erro crítico no setup do banco: {e}")
        raise

if __name__ == "__main__":
    initialize_system()
