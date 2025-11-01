"""
settings_dynamic.py
----------------------------------------------------
Camada de configuração dinâmica:
- Lê defaults do .env (os.environ)
- Sobrepõe com valores do MariaDB (estado atual)
- Sobrepõe com valores do Redis (runtime)
- Expõe propriedades tipadas p/ o restante do app
- Registra histórico de mudanças em settings_history (auditoria)

Tabelas criadas:
  settings_current(sk PK, svalue TEXT, updated_at TIMESTAMP)
  settings_history(id, sk, svalue, source, actor, created_at)

Uso:
  from .settings_dynamic import settings
  models = settings.CANDIDATE_MODELS_LIST
  settings.set("TEMPERATURE_DEFAULT", 0.4, actor="dash", source="ui")
"""

from __future__ import annotations
import os
import re
import json
import logging
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from .utils.redis_client import get_redis

logger = logging.getLogger(__name__)


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_list(v: Any, default: Optional[List[str]] = None) -> List[str]:
    if default is None:
        default = []
    if v is None:
        return default
    if isinstance(v, list):
        return v
    s = str(v).strip()
    # JSON list?
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    # Comma-separated
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    # Single value
    return [s] if s else default


def _serialize(value: Any) -> str:
    if isinstance(value, (dict, list, bool, int, float)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _deserialize(raw: Optional[str]) -> Any:
    if raw is None:
        return None
    s = raw.strip()
    # JSON?
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            return s
    # bool?
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    # number?
    if re.match(r"^-?\d+(\.\d+)?$", s):
        return float(s) if "." in s else int(s)
    return s


class DynamicSettings:
    """
    Lê configurações em camadas: Redis > DB (settings_current) > os.environ
    Grava alterações em Redis + DB (e histórico em settings_history).
    """

    # Chaves "oficiais" mais usadas pela aplicação
    KNOWN_KEYS = {
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "EMBED_MODEL",
        "CANDIDATE_MODELS_LIST",
        "JUDGE_MODELS",
        "JUDGES_MODE",
        "JUDGE_USE_RAG",
        "ENABLE_RAG_FOR_JUDGES",
        "BANDIT_STATE_PATH",
        "PROMETHEUS_PORT",
        "OLLAMA_SERVERS",
        "OLLAMA_MAX_PARALLEL",
        "OLLAMA_TIMEOUT",
        "DB_HOST", "DB_USER", "DB_PASS", "DB_NAME",
        "REDIS_HOST", "REDIS_PORT", "REDIS_DB",
        "CACHE_TTL",
        "TEMPERATURE_DEFAULT",
        "MAX_TOKENS_DEFAULT",
        "ADMIN_TOKEN",
    }

    def __init__(self) -> None:
        self.env = os.environ
        self.logger = logging.getLogger("settings_dynamic")
        self._engine = self._make_engine()
        self._ensure_tables()
        self._db_cache: Dict[str, str] = self._load_db_settings()
        self._redis = get_redis()

    # ---------- Infra DB ----------
    def _make_engine(self):
        host = self.env.get("DB_HOST", "mariadb")
        user = self.env.get("DB_USER", "router_user")
        pwd = self.env.get("DB_PASS", "router_pass")
        name = self.env.get("DB_NAME", "routerdb")
        url = f"mysql+pymysql://{user}:{pwd}@{host}:3306/{name}"
        return create_engine(url, pool_pre_ping=True, pool_recycle=3600)

    def _ensure_tables(self) -> None:
        ddl_current = """
        CREATE TABLE IF NOT EXISTS settings_current (
            sk VARCHAR(128) PRIMARY KEY,
            svalue TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP
        );
        """
        ddl_history = """
        CREATE TABLE IF NOT EXISTS settings_history (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            sk VARCHAR(128) NOT NULL,
            svalue TEXT NOT NULL,
            source VARCHAR(32) DEFAULT 'api',
            actor VARCHAR(128) NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_sk_created (sk, created_at)
        );
        """
        try:
            with self._engine.begin() as conn:
                conn.execute(text(ddl_current))
                conn.execute(text(ddl_history))
        except SQLAlchemyError as e:
            self.logger.warning(f"[settings] Falha ao criar tabelas de settings: {e}")

    def _load_db_settings(self) -> Dict[str, str]:
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(text("SELECT sk, svalue FROM settings_current")).fetchall()
                return {str(r[0]): str(r[1]) for r in rows}
        except SQLAlchemyError as e:
            self.logger.warning(f"[settings] Falha ao carregar settings do DB: {e}")
            return {}

    # ---------- Get/Set ----------
    def get(self, key: str, default: Any = None) -> Any:
        # 1) Redis
        try:
            if self._redis:
                raw = self._redis.get(f"cfg:{key}")
                if raw is not None:
                    return _deserialize(raw)
        except Exception as e:
            self.logger.warning(f"[settings] Falha ao consultar Redis para {key}: {e}")

        # 2) DB cache
        raw_db = self._db_cache.get(key)
        if raw_db is not None:
            return _deserialize(raw_db)

        # 3) .env
        raw_env = self.env.get(key)
        if raw_env is not None:
            return _deserialize(raw_env)

        return default

    def set(self, key: str, value: Any, *, actor: str = "system", source: str = "api") -> None:
        sval = _serialize(value)

        # Redis
        try:
            if self._redis:
                self._redis.set(f"cfg:{key}", sval)
        except Exception as e:
            self.logger.warning(f"[settings] Falha ao gravar Redis ({key}): {e}")

        # DB + histórico
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO settings_current (sk, svalue)
                        VALUES (:k, :v)
                        ON DUPLICATE KEY UPDATE svalue = :v, updated_at = CURRENT_TIMESTAMP
                    """),
                    {"k": key, "v": sval},
                )
                conn.execute(
                    text("""
                        INSERT INTO settings_history (sk, svalue, source, actor)
                        VALUES (:k, :v, :src, :act)
                    """),
                    {"k": key, "v": sval, "src": source, "act": actor},
                )
            self._db_cache[key] = sval
        except SQLAlchemyError as e:
            self.logger.warning(f"[settings] Falha ao persistir setting {key}: {e}")

    def snapshot(self, only_known: bool = True) -> Dict[str, Any]:
        keys = self.KNOWN_KEYS if only_known else set(self.env.keys()) | set(self._db_cache.keys())
        out: Dict[str, Any] = {}
        for k in sorted(keys):
            out[k] = self.get(k, None)
        return out

    # ---------- Propriedades tipadas ----------
    @property
    def OLLAMA_BASE_URL(self) -> str:
        return str(self.get("OLLAMA_BASE_URL", "http://ollama:11434"))

    @property
    def OLLAMA_MODEL(self) -> str:
        return str(self.get("OLLAMA_MODEL", "ollama/gemma3:4b-it-qat"))

    @property
    def EMBED_MODEL(self) -> str:
        return str(self.get("EMBED_MODEL", "nomic-embed-text"))

    @property
    def CANDIDATE_MODELS_LIST(self) -> List[str]:
        default = _as_list(self.env.get("CANDIDATE_MODELS_LIST", ""), [])
        return _as_list(self.get("CANDIDATE_MODELS_LIST", default), default)

    @property
    def JUDGE_MODELS(self) -> List[str]:
        default = _as_list(self.env.get("JUDGE_MODELS", ""), [])
        return _as_list(self.get("JUDGE_MODELS", default), default)

    @property
    def JUDGES_MODE(self) -> str:
        return str(self.get("JUDGES_MODE", "hybrid")).lower()

    @property
    def JUDGE_USE_RAG(self) -> bool:
        return _as_bool(self.get("JUDGE_USE_RAG", True), True)

    @property
    def ENABLE_RAG_FOR_JUDGES(self) -> bool:
        return _as_bool(self.get("ENABLE_RAG_FOR_JUDGES", True), True)

    @property
    def BANDIT_STATE_PATH(self) -> str:
        return str(self.get("BANDIT_STATE_PATH", "/app/state/bandit.json"))

    @property
    def PROMETHEUS_PORT(self) -> int:
        return _as_int(self.get("PROMETHEUS_PORT", 9090), 9090)

    @property
    def OLLAMA_SERVERS(self) -> str:
        return str(self.get("OLLAMA_SERVERS", self.OLLAMA_BASE_URL))

    @property
    def OLLAMA_MAX_PARALLEL(self) -> int:
        return _as_int(self.get("OLLAMA_MAX_PARALLEL", 2), 2)

    @property
    def OLLAMA_TIMEOUT(self) -> int:
        return _as_int(self.get("OLLAMA_TIMEOUT", 90), 90)

    @property
    def DB_HOST(self) -> str:
        return str(self.get("DB_HOST", "mariadb"))

    @property
    def DB_USER(self) -> str:
        return str(self.get("DB_USER", "router_user"))

    @property
    def DB_PASS(self) -> str:
        return str(self.get("DB_PASS", "router_pass"))

    @property
    def DB_NAME(self) -> str:
        return str(self.get("DB_NAME", "routerdb"))

    @property
    def REDIS_HOST(self) -> str:
        return str(self.get("REDIS_HOST", "redis"))

    @property
    def REDIS_PORT(self) -> int:
        return _as_int(self.get("REDIS_PORT", 6379), 6379)

    @property
    def REDIS_DB(self) -> int:
        return _as_int(self.get("REDIS_DB", 0), 0)

    @property
    def CACHE_TTL(self) -> int:
        return _as_int(self.get("CACHE_TTL", 86400), 86400)

    @property
    def TEMPERATURE_DEFAULT(self) -> float:
        return _as_float(self.get("TEMPERATURE_DEFAULT", 0.5), 0.5)

    @property
    def MAX_TOKENS_DEFAULT(self) -> int:
        return _as_int(self.get("MAX_TOKENS_DEFAULT", 1024), 1024)

    @property
    def ADMIN_TOKEN(self) -> str:
        return str(self.get("ADMIN_TOKEN", "changeme-please"))


# Instância única
settings = DynamicSettings()
