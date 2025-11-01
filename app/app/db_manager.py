# db_manager.py
"""
db_manager.py
----------------------------------------------------
Gerencia a inicialização completa do banco de dados e integração com Redis.
- Cria e migra todas as tabelas necessárias para o funcionamento da aplicação.
- Executa init_db.sql, se existir.
- Popular Redis com variáveis críticas (modelos, epsilon, etc.).
"""

import os
import json
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from redis import Redis

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ============================================================
# ⚙️ Configurações do banco e Redis
# ============================================================

DB_HOST = os.getenv("DB_HOST", "mariadb")
DB_USER = os.getenv("DB_USER", "router_user")
DB_PASS = os.getenv("DB_PASS", "router_pass")
DB_NAME = os.getenv("DB_NAME", "routerdb")
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASS = os.getenv("REDIS_PASS", "SenhaForte")

engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)
redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS, decode_responses=True)

# ============================================================
# 🧱 Tabelas da aplicação
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
    # Métricas agregadas por modelo
    "model_metrics": """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            model VARCHAR(255) NOT NULL,
            avg_latency FLOAT DEFAULT 0,
            avg_quality FLOAT DEFAULT 0,
            avg_cost FLOAT DEFAULT 0,
            total_queries INT DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
                ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uq_model (model)
        );
    """,
    # Log detalhado de consultas
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
    # Cache semântico (para evitar reprocessamento)
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
}

# ============================================================
# 🗂 Execução do init_db.sql (opcional)
# ============================================================

def _execute_init_sql():
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
    """Cria ou atualiza todas as tabelas do sistema."""
    logger.info("[db_manager] Verificando e criando tabelas necessárias...")
    try:
        with engine.begin() as conn:
            for name, ddl in TABLES_DDL.items():
                conn.execute(text(ddl))
                logger.info(f"✅ Tabela verificada/criada: {name}")
    except SQLAlchemyError as e:
        logger.error(f"[db_manager] Erro ao criar tabelas: {e}")

# ============================================================
# 🧩 Inicialização do Redis
# ============================================================

def initialize_redis():
    """Garante que chaves críticas existam no Redis."""
    try:
        # Modelos candidatos e juízes
        candidate_models = os.getenv("CANDIDATE_MODELS_LIST", "").split(",")
        judge_models = os.getenv("JUDGE_MODELS", "").split(",")
        candidate_models = [m.strip() for m in candidate_models if m.strip()]
        judge_models = [m.strip() for m in judge_models if m.strip()]

        # Fallback se não houver lista
        if not candidate_models:
            candidate_models = ["gpt-4o-mini", "mistral", "llama3.1:8b", "phi3.5"]
        if not judge_models:
            judge_models = ["gpt-4.1-mini"]

        # Escreve no Redis (JSON)
        redis_client.set("router:candidate_models", json.dumps(candidate_models))
        redis_client.set("router:judge_models", json.dumps(judge_models))
        redis_client.set("router:bandit:epsilon", os.getenv("BANDIT_EPSILON", "0.15"))

        redis_client.set("router:bandit:ctx_version", "v2.0")
        redis_client.set("router:last_migration", "ok")

        logger.info(f"[redis] Modelos candidatos definidos: {candidate_models}")
        logger.info(f"[redis] Juízes definidos: {judge_models}")
        logger.info(f"[redis] Epsilon padrão configurado.")
    except Exception as e:
        logger.error(f"[db_manager] Falha ao inicializar Redis: {e}")

# ============================================================
# 🚀 Inicialização completa
# ============================================================

def initialize_system():
    """Inicializa todo o backend da aplicação."""
    logger.info("🚀 Iniciando setup completo do sistema...")
    initialize_tables()
    _execute_init_sql()
    initialize_redis()
    logger.info("✅ Banco e Redis inicializados com sucesso.")

# ============================================================
# 🎯 Execução direta
# ============================================================

if __name__ == "__main__":
    initialize_system()
    logger.info("[db_manager] Inicialização concluída.")
