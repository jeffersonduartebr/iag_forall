# -*- coding: utf-8 -*-
"""
settings_dynamic.py
------------------------------------------------------
Carrega configurações com fallback:

1) Redis
2) DB
3) .env
4) Defaults

Inclui:
- Configs do roteador
- Redis
- DB
- Embeddings
- Centróides online
- Juízes automáticos (JUDGES_MODE)
"""

import os
import json
import logging
from sqlalchemy import create_engine, text
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

REDIS_PREFIX = "settings:"
rds = get_redis()

# Banco inicial (para leitura de settings)
DB_HOST_ENV = os.getenv("DB_HOST", "mariadb")
DB_USER_ENV = os.getenv("DB_USER", "router_user")
DB_PASS_ENV = os.getenv("DB_PASS", "router_pass")
DB_NAME_ENV = os.getenv("DB_NAME", "routerdb")
DB_URL = f"mysql+pymysql://{DB_USER_ENV}:{DB_PASS_ENV}@{DB_HOST_ENV}:3306/{DB_NAME_ENV}"

engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

DDL = """
CREATE TABLE IF NOT EXISTS settings_dynamic (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    setting_key VARCHAR(512) NOT NULL UNIQUE,
    setting_value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
);
"""

try:
    with engine.begin() as conn:
        conn.execute(text(DDL))
except Exception as e:
    logger.warning(f"[settings_dynamic] Falha ao criar tabela: {e}")

# ============================================================
# Utilitários
# ============================================================

def _get_from_redis(key: str):
    try:
        if rds:
            v = rds.get(f"{REDIS_PREFIX}{key}")
            if v:
                return v.decode()
    except Exception:
        pass
    return None

def _get_from_db(key: str):
    try:
        with engine.connect() as conn:
            r = conn.execute(
                text("SELECT setting_value FROM settings_dynamic WHERE setting_key=:k LIMIT 1"),
                {"k": key}
            ).fetchone()
        if r:
            return r[0]
    except Exception:
        pass
    return None

def _set_to_redis(key: str, val: str):
    try:
        if rds:
            rds.set(f"{REDIS_PREFIX}{key}", val)
    except Exception:
        pass

def set_setting(key: str, value: str):
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO settings_dynamic (setting_key, setting_value)
                    VALUES (:k, :v)
                    ON DUPLICATE KEY UPDATE setting_value = :v
                """),
                {"k": key, "v": value}
            )
        _set_to_redis(key, value)
    except Exception as e:
        logger.warning(f"[settings_dynamic] Falha ao atualizar {key}: {e}")

# ============================================================
# Classe principal
# ============================================================

class DynamicSettings:

    DEFAULTS = {
        # -------------------
        # Router core
        # -------------------
        "MAX_TOKENS_DEFAULT": "2000",
        "TEMPERATURE_DEFAULT": "0.55",
        "BANDIT_EPSILON": "0.12",
        "QUERY_LOG_RETENTION_DAYS": "7",

        # -------------------
        # Redis compat
        # -------------------
        "REDIS_HOST": "redis",
        "REDIS_PORT": "6379",
        "REDIS_DB": "0",
        "REDIS_PASSWORD": "",

        # -------------------
        # Embeddings
        # -------------------
        "EMBED_MODEL": "nomic-embed-text",
        "EMBED_PROVIDER": "ollama",
        "EMBED_DEVICE": "cpu",

        # -------------------
        # Centróides online
        # -------------------
        "CENTROIDS_DIM": "768",
        "CENTROIDS_K": "20",
        "CENTROIDS_MIN_SIM_CREATE": "0.35",
        "CENTROIDS_ENABLE_ONLINE": "1",
        "CENTROIDS_UPDATE_INTERVAL_S": "1800",
        "CENTROIDS_MIN_RECORDS_FOR_TRAIN": "50",
        "CENTROIDS_MAX_HISTORY": "50000",

        # -------------------
        # Judges (novo!)
        # -------------------
        "JUDGES_ENABLED": "1",
        "JUDGES_MODE": "hybrid",  # local | remote | hybrid
        "JUDGES_LOCAL_MODEL": "ollama/phi4:latest",
        "JUDGES_REMOTE_MODEL": "gpt-4o-mini",
        "JUDGES_TIMEOUT_S": "15",
        # -------------------
        # Ollama
        # -------------------
        "OLLAMA_BASE_URL": "http://ollama:11434",
        "OLLAMA_HOST": "http://ollama:11434",
        
    }

    def get(self, key: str, fallback=None):
        v = _get_from_redis(key)
        if v is not None:
            return v
        v = _get_from_db(key)
        if v is not None:
            return v
        v = os.getenv(key)
        if v is not None:
            return v
        return self.DEFAULTS.get(key, fallback)

    # -------------------------
    # DB configs
    # -------------------------
    @property
    def DB_HOST(self): return self.get("DB_HOST", DB_HOST_ENV)

    @property
    def DB_USER(self): return self.get("DB_USER", DB_USER_ENV)

    @property
    def DB_PASS(self): return self.get("DB_PASS", DB_PASS_ENV)

    @property
    def DB_NAME(self): return self.get("DB_NAME", DB_NAME_ENV)

    # -------------------------
    # Redis configs
    # -------------------------
    @property
    def REDIS_HOST(self): return self.get("REDIS_HOST")

    @property
    def REDIS_PORT(self): return int(self.get("REDIS_PORT"))

    @property
    def REDIS_DB(self): return int(self.get("REDIS_DB"))

    @property
    def REDIS_PASSWORD(self): return self.get("REDIS_PASSWORD")

    # -------------------------
    # Models
    # -------------------------
    @property
    def CANDIDATE_MODELS_LIST(self):
        try:
            raw = self.get("CANDIDATE_MODELS_LIST", "[]")
            return json.loads(raw)
        except Exception:
            return []

    # -------------------------
    # Router core
    # -------------------------
    @property
    def MAX_TOKENS_DEFAULT(self): return int(self.get("MAX_TOKENS_DEFAULT"))

    @property
    def TEMPERATURE_DEFAULT(self): return float(self.get("TEMPERATURE_DEFAULT"))

    @property
    def BANDIT_EPSILON(self): return float(self.get("BANDIT_EPSILON"))

    @property
    def QUERY_LOG_RETENTION_DAYS(self): return int(self.get("QUERY_LOG_RETENTION_DAYS"))

    # -------------------------
    # Embeddings
    # -------------------------
    @property
    def EMBED_MODEL(self): return self.get("EMBED_MODEL")

    @property
    def EMBED_PROVIDER(self): return self.get("EMBED_PROVIDER")

    @property
    def EMBED_DEVICE(self): return self.get("EMBED_DEVICE")

    # -------------------------
    # Judges
    # -------------------------
    @property
    def JUDGES_ENABLED(self): return self.get("JUDGES_ENABLED") == "1"

    @property
    def JUDGES_MODE(self): return self.get("JUDGES_MODE")

    @property
    def JUDGES_LOCAL_MODEL(self): return self.get("JUDGES_LOCAL_MODEL")

    @property
    def JUDGES_REMOTE_MODEL(self): return self.get("JUDGES_REMOTE_MODEL")

    @property
    def JUDGES_TIMEOUT_S(self): return int(self.get("JUDGES_TIMEOUT_S"))

    # -------------------------
    # Centróides online
    # -------------------------
    @property
    def CENTROIDS_DIM(self): return int(self.get("CENTROIDS_DIM"))

    @property
    def CENTROIDS_K(self): return int(self.get("CENTROIDS_K"))

    @property
    def CENTROIDS_MIN_SIM_CREATE(self): return float(self.get("CENTROIDS_MIN_SIM_CREATE"))

    @property
    def CENTROIDS_ENABLE_ONLINE(self): return self.get("CENTROIDS_ENABLE_ONLINE") == "1"

    @property
    def CENTROIDS_UPDATE_INTERVAL_S(self): return int(self.get("CENTROIDS_UPDATE_INTERVAL_S"))

    @property
    def CENTROIDS_MIN_RECORDS_FOR_TRAIN(self): return int(self.get("CENTROIDS_MIN_RECORDS_FOR_TRAIN"))

    @property
    def CENTROIDS_MAX_HISTORY(self): return int(self.get("CENTROIDS_MAX_HISTORY"))
    @property
    def OLLAMA_HOST(self):
        return self.get("OLLAMA_HOST") or "http://ollama:11434"

    @property
    def OLLAMA_BASE_URL(self):
        # alias para compatibilidade retroativa
        return self.get("OLLAMA_BASE_URL") or self.OLLAMA_HOST


    # -------------------------
    # Snapshot para /health
    # -------------------------
    def snapshot(self, only_known=False):
        out = {}
        keys = list(self.DEFAULTS.keys())

        if not only_known:
            keys += ["DB_HOST", "DB_USER", "DB_NAME"]

        for k in keys:
            try:
                out[k] = self.get(k)
            except Exception:
                out[k] = None
        return out


settings = DynamicSettings()
