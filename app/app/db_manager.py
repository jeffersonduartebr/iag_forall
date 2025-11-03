# -*- coding: utf-8 -*-
"""
db_manager.py
----------------------------------------------------
Gerencia a inicialização do banco de dados (criação de tabelas).
- Cria e migra todas as tabelas de DADOS necessárias.
- Executa init_db.sql, se existir.

NOTA: Este módulo NÃO gerencia mais a configuração do Redis.
Isso é feito pelo 'settings_dynamic.py' e pelos endpoints de admin.
"""

import os
import json
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from redis import Redis

# ✅ Importa settings centralizado
try:
    from app.settings_dynamic import settings
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from app.settings_dynamic import settings


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ============================================================
# ⚙️ Configurações de banco e Redis
# ============================================================

DB_HOST = settings.DB_HOST
DB_USER = settings.DB_USER
DB_PASS = settings.DB_PASS
DB_NAME = settings.DB_NAME
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"

REDIS_HOST = settings.REDIS_HOST
REDIS_PORT = settings.REDIS_PORT
REDIS_PASS = settings.get("REDIS_PASS", "SenhaForte")

engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)
redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS, decode_responses=True)

# ============================================================
# 🧱 Tabelas da aplicação (schemas de DADOS)
# ============================================================

TABLES_DDL = {
    # Histórico das médias exponenciais
    "ema_history": """
        CREATE TABLE IF NOT EXISTS ema_history (
            model VARCHAR(255) PRIMARY KEY,
            ema_latency FLOAT NOT NULL,
            ema_quality FLOAT NOT NULL,
            ema_cost FLOAT NOT NULL,
            updates INT DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
                ON UPDATE CURRENT_TIMESTAMP
        );
    """,

    # Log de atualizações de EMA
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
        );
    """,

    # Estatísticas contextuais do Bandit
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
        ) ENGINE=InnoDB;
    """,

    # Log detalhado de consultas (usado por NSGA e métricas)
    "query_log": """
        CREATE TABLE IF NOT EXISTS query_log (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            query_text TEXT NOT NULL,
            chosen_model VARCHAR(255) NOT NULL,
            answer TEXT,
            quality FLOAT,
            latency_s FLOAT,
            cost_per_1k FLOAT,
            reward FLOAT,
            context_label VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_created_at (created_at)
        );
    """,

    # Cache semântico (dados auxiliares)
    "semantic_cache": """
        CREATE TABLE IF NOT EXISTS semantic_cache (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            query_hash VARCHAR(64) UNIQUE,
            query_text TEXT,
            context_category VARCHAR(50),
            answer TEXT,
            model_used VARCHAR(255),
            embedding LONGBLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,

    # Log dos juízes (auditoria de fallback e scores)
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
        );
    """,

    # Métricas detalhadas de desempenho dos modelos
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
        );
    """
}

# ============================================================
# 🗂 Execução opcional de init_db.sql
# ============================================================

def _execute_init_sql():
    """Executa o script init_db.sql, se existir no diretório /db."""
    sql_path = os.path.join(os.getcwd(), "db", "init_db.sql")
    if not os.path.exists(sql_path):
        logger.info("[db_manager] Nenhum init_db.sql encontrado — prosseguindo.")
        return

    try:
        with open(sql_path, "r", encoding="utf-8") as f:
            sql_script = f.read()
        with engine.begin() as conn:
            for stmt in sql_script.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
        logger.info("[db_manager] init_db.sql executado com sucesso.")
    except Exception as e:
        logger.error(f"[db_manager] Erro ao executar init_db.sql: {e}")

# ============================================================
# 🔧 Criação automática de tabelas
# ============================================================

def initialize_tables():
    """Cria ou atualiza todas as tabelas necessárias."""
    logger.info("[db_manager] Verificando/criando tabelas de dados...")
    try:
        with engine.begin() as conn:
            for name, ddl in TABLES_DDL.items():
                try:
                    conn.execute(text(ddl))
                    logger.info(f"✅ Tabela verificada/criada: {name}")
                except SQLAlchemyError as e:
                    logger.error(f"[db_manager] Falha ao criar tabela {name}: {e}")
    except SQLAlchemyError as e:
        logger.error(f"[db_manager] Erro geral ao criar tabelas: {e}")

# ============================================================
# 🚀 Inicialização completa
# ============================================================

def initialize_system():
    """Inicializa todas as tabelas e scripts opcionais."""
    logger.info("🚀 Iniciando setup do banco de dados...")
    initialize_tables()
    _execute_init_sql()
    logger.info("✅ Banco de dados inicializado com sucesso.")

# ============================================================
# 🎯 Execução direta
# ============================================================

if __name__ == "__main__":
    initialize_system()
    logger.info("[db_manager] Inicialização concluída.")
