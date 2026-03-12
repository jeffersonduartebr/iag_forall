# -*- coding: utf-8 -*-
"""
settings_dynamic.py (VERSÃO FINAL: Com Configuração de Amostragem)
--------------------------------------------------------------------------
Carrega configurações com fallback em camadas.
"""

from __future__ import annotations

import os
import time
import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, List

from sqlalchemy import text
from app.utils.redis_client import get_redis_async_safe, ensure_redis_connected
from .config.settings_catalog import (
    SETTING_METADATA,
    SETTINGS_DEFAULTS,
    is_known_setting,
    is_runtime_mutable,
    known_setting_keys,
)
from .config.settings_sources import decode_redis_value, resolve_setting_value
from .config.settings_types import as_bool, as_float, as_int, as_list

logger = logging.getLogger(__name__)

# ============================================================
# Redis / Banco básicos
# ============================================================

REDIS_PREFIX = "settings:"
REDIS_RELOAD_CHANNEL = "settings:reload"

def _get_rds():
    """Executa a responsabilidade descrita por este método.

    Returns:
        Valor produzido pela execução.
    """
    return get_redis_async_safe() or ensure_redis_connected(max_wait_s=0.0, min_retry_interval_s=2.0)

# Environment variable defaults (used by db.py)
DB_HOST_ENV = os.getenv("DB_HOST", "mariadb")
DB_USER_ENV = os.getenv("DB_USER", "router_user")
DB_PASS_ENV = os.getenv("DB_PASS", "")
DB_NAME_ENV = os.getenv("DB_NAME", "routerdb")
DB_PORT_ENV = int(os.getenv("DB_PORT", "3306"))


def _get_settings_engine():
    """
    Get database engine for settings.
    Uses lazy import to avoid circular dependencies during startup.
    """
    try:
        from app.db import get_engine
        return get_engine()
    except Exception:
        # Fallback: create a temporary engine for bootstrap
        from sqlalchemy import create_engine
        db_url = f"mysql+pymysql://{DB_USER_ENV}:{DB_PASS_ENV}@{DB_HOST_ENV}:{DB_PORT_ENV}/{DB_NAME_ENV}"
        return create_engine(db_url, pool_pre_ping=True, pool_recycle=300)


# Legacy alias for backward compatibility
engine = property(lambda self: _get_settings_engine())


class _EngineProxy:
    """Proxy to centralized engine for backward compatibility."""

    def begin(self):
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return _get_settings_engine().begin()

    def connect(self):
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return _get_settings_engine().connect()

    @property
    def pool(self):
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return _get_settings_engine().pool


engine = _EngineProxy()

DDL = """
CREATE TABLE IF NOT EXISTS settings_dynamic (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    setting_key VARCHAR(512) NOT NULL UNIQUE,
    setting_value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;
"""

try:
    with engine.begin() as conn:
        conn.execute(text(DDL))
except Exception as e:
    logger.warning(f"[settings_dynamic] Falha ao criar tabela: {e}")


# ============================================================
# LRU Cache interno com TTL
# ============================================================

class LRUCache:
    """Define responsabilidades de estado e comportamento."""
    def __init__(self, maxsize: int = 512, ttl_s: int = 30):
        """Executa a responsabilidade descrita por este método.

        Args:
            maxsize: Parâmetro de entrada.
            ttl_s: Parâmetro de entrada.

        Returns:
            Valor produzido pela execução.
        """
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Executa a responsabilidade descrita por este método.

        Args:
            key: Parâmetro de entrada.

        Returns:
            Valor produzido pela execução.
        """
        now = time.time()
        with self._lock:
            if key not in self._data:
                return None
            value, ts = self._data[key]
            if self.ttl_s > 0 and (now - ts) > self.ttl_s:
                try:
                    del self._data[key]
                except KeyError:
                    pass
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """Executa a responsabilidade descrita por este método.

        Args:
            key: Parâmetro de entrada.
            value: Parâmetro de entrada.

        Returns:
            Valor produzido pela execução.
        """
        now = time.time()
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (value, now)
            if len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        with self._lock:
            self._data.clear()


SETTINGS_CACHE_SIZE = int(os.getenv("SETTINGS_CACHE_SIZE", "2000"))  # Optimized for high-capacity
SETTINGS_CACHE_TTL_S = int(os.getenv("SETTINGS_CACHE_TTL_S", "300"))  # 5 min - reduces Redis/DB lookups by ~80%

_lru = LRUCache(maxsize=SETTINGS_CACHE_SIZE, ttl_s=SETTINGS_CACHE_TTL_S)


def _invalidate_cache():
    """Executa a responsabilidade descrita por este método.

    Returns:
        Valor produzido pela execução.
    """
    _lru.clear()
    logger.info("[settings_dynamic] Cache LRU invalidado.")


# ============================================================
# Auxiliares de acesso Redis / DB
# ============================================================

def _get_from_redis(key: str) -> Optional[str]:
    """Executa a responsabilidade descrita por este método.

    Args:
        key: Parâmetro de entrada.

    Returns:
        Valor produzido pela execução.
    """
    rds = _get_rds()
    if not rds:
        return None
    try:
        return decode_redis_value(rds.get(f"{REDIS_PREFIX}{key}"))
    except Exception:
        pass
    return None


def _get_from_db(key: str) -> Optional[str]:
    """Executa a responsabilidade descrita por este método.

    Args:
        key: Parâmetro de entrada.

    Returns:
        Valor produzido pela execução.
    """
    try:
        with engine.connect() as conn:
            r = conn.execute(
                text(
                    "SELECT setting_value FROM settings_dynamic "
                    "WHERE setting_key=:k LIMIT 1"
                ),
                {"k": key},
            ).fetchone()
        if r:
            return r[0]
    except Exception:
        pass
    return None


def _set_to_redis(key: str, val: str):
    """Executa a responsabilidade descrita por este método.

    Args:
        key: Parâmetro de entrada.
        val: Parâmetro de entrada.

    Returns:
        Valor produzido pela execução.
    """
    rds = _get_rds()
    if not rds:
        return
    try:
        rds.set(f"{REDIS_PREFIX}{key}", val)
    except Exception as e:
        logger.debug(f"[settings_dynamic] Falha ao gravar Redis para {key}: {e}")


def _load_json_list(raw: Optional[str]) -> List[str]:
    """Executa a responsabilidade descrita por este método.

    Args:
        raw: Parâmetro de entrada.

    Returns:
        Valor produzido pela execução.
    """
    return as_list(raw)


# ============================================================
# Classe Principal (Integrada)
# ============================================================

class DynamicSettings:

    """Define responsabilidades de estado e comportamento."""
    DEFAULTS: Dict[str, str] = SETTINGS_DEFAULTS
    METADATA: Dict[str, Dict[str, str]] = SETTING_METADATA

    def get(self, key: str, fallback: Any = None) -> Any:
        """Executa a responsabilidade descrita por este método.

        Args:
            key: Parâmetro de entrada.
            fallback: Parâmetro de entrada.

        Returns:
            Valor produzido pela execução.
        """
        return resolve_setting_value(
            key=key,
            fallback=fallback,
            defaults=self.DEFAULTS,
            cache_get=_lru.get,
            cache_set=_lru.set,
            redis_get=_get_from_redis,
            db_get=_get_from_db,
            env_get=os.getenv,
        )

    def _get_str(self, key: str, default: str = "") -> str:
        """Get a setting coerced to string."""
        value = self.get(key, default)
        return default if value is None else str(value)

    def _get_int(self, key: str, default: int) -> int:
        """Get a setting coerced to int."""
        return as_int(self.get(key, default), default)

    def _get_float(self, key: str, default: float) -> float:
        """Get a setting coerced to float."""
        return as_float(self.get(key, default), default)

    def _get_bool(self, key: str, default: bool = False) -> bool:
        """Get a setting coerced to bool."""
        return as_bool(self.get(key, "1" if default else "0"), default)

    def _get_list(self, key: str, default: str = "[]") -> List[str]:
        """Get a setting coerced to string list."""
        return as_list(self.get(key, default))

    def set(self, key: str, value: str, actor: str = "system", source: str = "internal") -> None:
        """Executa a responsabilidade descrita por este método.

        Args:
            key: Parâmetro de entrada.
            value: Parâmetro de entrada.
            actor: Parâmetro de entrada.
            source: Parâmetro de entrada.

        Returns:
            Valor produzido pela execução.
        """
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO settings_dynamic (setting_key, setting_value)
                        VALUES (:k, :v)
                        ON DUPLICATE KEY UPDATE setting_value = :v
                    """),
                    {"k": key, "v": value},
                )
        except Exception as e:
            logger.warning(f"Falha ao gravar DB ({key}): {e}")
        _set_to_redis(key, value)
        _invalidate_cache()
        rds = _get_rds()
        if rds:
            try:
                rds.publish(REDIS_RELOAD_CHANNEL, key)
            except Exception:
                pass
        logger.info(f"[settings] '{key}' atualizado por {actor} via {source}.")

    def snapshot(self, only_known: bool = False) -> Dict[str, Any]:
        """Executa a responsabilidade descrita por este método.

        Args:
            only_known: Parâmetro de entrada.

        Returns:
            Valor produzido pela execução.
        """
        keys = known_setting_keys() if only_known else known_setting_keys() + [
            "DB_HOST", "DB_USER", "DB_NAME", "DB_PORT",
            "OLLAMA_HOST", "OLLAMA_BASE_URL",
        ]
        out = {}
        for k in keys:
            try:
                out[k] = self.get(k)
            except Exception:
                out[k] = None
        return out

    def keys(self, domain: Optional[str] = None) -> List[str]:
        """Return known setting keys, optionally filtered by catalog domain."""
        if not domain:
            return known_setting_keys()
        return [key for key, meta in self.METADATA.items() if meta.get("domain") == domain]

    def metadata(self, key: str) -> Dict[str, str]:
        """Return catalog metadata for one known setting."""
        return dict(self.METADATA.get(key, {}))

    def can_update_runtime(self, key: str) -> bool:
        """Return whether one setting can be updated dynamically via admin API."""
        return is_runtime_mutable(key)

    def validate_runtime_updates(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an admin update payload against the runtime mutability catalog."""
        unknown = [key for key in payload if not is_known_setting(key)]
        restart_required = [key for key in payload if is_known_setting(key) and not is_runtime_mutable(key)]
        runtime_safe = [key for key in payload if key not in unknown and key not in restart_required]
        return {
            "runtime_safe": runtime_safe,
            "requires_restart": restart_required,
            "unknown": unknown,
        }

    # -------------------------
    # Propriedades Tipadas
    # -------------------------
    @property
    def CANDIDATE_MODELS_LIST(self) -> List[str]:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return self._get_list("CANDIDATE_MODELS_LIST")

    @property
    def CANDIDATE_VISION_MODELS_LIST(self) -> List[str]:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return self._get_list("CANDIDATE_VISION_MODELS_LIST")

    @property
    def CANDIDATE_MULTIMODAL_MODELS_LIST(self) -> List[str]:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return self._get_list("CANDIDATE_MULTIMODAL_MODELS_LIST")

    @property
    def VLM_OLLAMA_MODELS(self) -> List[str]:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return self._get_list("VLM_OLLAMA_MODELS")
    
    @property
    def JUDGE_MODELS(self) -> List[str]:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return self._get_list("JUDGE_MODELS")

    # Embeddings
    @property
    def EMBED_TEXT_MODEL(self) -> str:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return self._get_str("EMBED_TEXT_MODEL", "nomic-embed-text")

    @property
    def TEXT_EMBEDDING_MODEL(self) -> str:
        """Obtém o valor da configuração `TEXT_EMBEDDING_MODEL`."""
        return self.get("TEXT_EMBEDDING_MODEL")
    @property
    def IMAGE_EMBEDDING_MODEL(self) -> str:
        """Obtém o valor da configuração `IMAGE_EMBEDDING_MODEL`."""
        return self.get("IMAGE_EMBEDDING_MODEL")
    @property
    def MULTIMODAL_EMBEDDING_MODEL(self) -> str:
        """Obtém o valor da configuração `MULTIMODAL_EMBEDDING_MODEL`."""
        return self.get("MULTIMODAL_EMBEDDING_MODEL")
    @property
    def EMBED_MODEL(self) -> str:
        """Obtém o valor da configuração `EMBED_MODEL`."""
        return self.get("EMBED_MODEL")
    @property
    def EMBED_PROVIDER(self) -> str:
        """Obtém o valor da configuração `EMBED_PROVIDER`."""
        return self.get("EMBED_PROVIDER")
    @property
    def EMBED_DEVICE(self) -> str:
        """Obtém o valor da configuração `EMBED_DEVICE`."""
        return self.get("EMBED_DEVICE")

    # Router Params
    @property
    def MAX_TOKENS_DEFAULT(self) -> int:
        """Obtém o valor da configuração `MAX_TOKENS_DEFAULT`."""
        return self._get_int("MAX_TOKENS_DEFAULT", 2000)
    @property
    def TEMPERATURE_DEFAULT(self) -> float:
        """Obtém o valor da configuração `TEMPERATURE_DEFAULT`."""
        return self._get_float("TEMPERATURE_DEFAULT", 0.5)
    @property
    def BANDIT_EPSILON(self) -> float:
        """Obtém o valor da configuração `BANDIT_EPSILON`."""
        return self._get_float("BANDIT_EPSILON", 0.12)
    @property
    def QUERY_LOG_RETENTION_DAYS(self) -> int:
        """Obtém o valor da configuração `QUERY_LOG_RETENTION_DAYS`."""
        return self._get_int("QUERY_LOG_RETENTION_DAYS", 7)
    
    # Banco / Infra
    @property
    def REDIS_HOST(self) -> str:
        """Obtém o valor da configuração `REDIS_HOST`."""
        return self.get("REDIS_HOST")
    @property
    def REDIS_PORT(self) -> int:
        """Obtém o valor da configuração `REDIS_PORT`."""
        return self._get_int("REDIS_PORT", 6379)
    @property
    def REDIS_DB(self) -> int:
        """Obtém o valor da configuração `REDIS_DB`."""
        return self._get_int("REDIS_DB", 0)
    @property
    def REDIS_PASSWORD(self) -> str:
        """Obtém o valor da configuração `REDIS_PASSWORD`."""
        return self.get("REDIS_PASSWORD")

    @property
    def DB_HOST(self) -> str:
        """Obtém o valor da configuração `DB_HOST`."""
        return self.get("DB_HOST", DB_HOST_ENV)
    @property
    def DB_PORT(self) -> int:
        """Obtém o valor da configuração `DB_PORT`."""
        return self._get_int("DB_PORT", DB_PORT_ENV)
    @property
    def DB_USER(self) -> str:
        """Obtém o valor da configuração `DB_USER`."""
        return self.get("DB_USER", DB_USER_ENV)
    @property
    def DB_PASS(self) -> str:
        """Obtém o valor da configuração `DB_PASS`."""
        return self.get("DB_PASS", DB_PASS_ENV)
    @property
    def DB_NAME(self) -> str:
        """Obtém o valor da configuração `DB_NAME`."""
        return self.get("DB_NAME", DB_NAME_ENV)
    
    @property
    def ADMIN_TOKEN(self) -> str:
        """Obtém o valor da configuração `ADMIN_TOKEN`."""
        return self.get("ADMIN_TOKEN", "")
    @property
    def ADMIN_TOKEN_PREVIOUS(self) -> str:
        """Obtém o valor da configuração `ADMIN_TOKEN_PREVIOUS`."""
        return self.get("ADMIN_TOKEN_PREVIOUS", "")

    # Judges
    @property
    def JUDGES_ENABLED(self) -> bool:
        """Obtém o valor da configuração `JUDGES_ENABLED`."""
        return self._get_bool("JUDGES_ENABLED", True)
    @property
    def JUDGES_MODE(self) -> str:
        """Obtém o valor da configuração `JUDGES_MODE`."""
        return self.get("JUDGES_MODE")
    @property
    def JUDGES_LOCAL_MODEL(self) -> str:
        """Obtém o valor da configuração `JUDGES_LOCAL_MODEL`."""
        return self.get("JUDGES_LOCAL_MODEL")
    @property
    def JUDGES_REMOTE_MODEL(self) -> str:
        """Obtém o valor da configuração `JUDGES_REMOTE_MODEL`."""
        return self.get("JUDGES_REMOTE_MODEL")
    @property
    def JUDGES_TIMEOUT_S(self) -> int:
        """Obtém o valor da configuração `JUDGES_TIMEOUT_S`."""
        return self._get_int("JUDGES_TIMEOUT_S", 15)
    @property
    def JUDGE_MIN_SAMPLE_RATE(self) -> float:
        """Obtém o valor da configuração `JUDGE_MIN_SAMPLE_RATE`."""
        return self._get_float("JUDGE_MIN_SAMPLE_RATE", 0.05)

    # Ollama
    @property
    def OLLAMA_HOST(self) -> str:
        """Obtém o valor da configuração `OLLAMA_HOST`."""
        return self.get("OLLAMA_HOST") or "http://ollama:11434"
    @property
    def OLLAMA_BASE_URL(self) -> str:
        """Obtém o valor da configuração `OLLAMA_BASE_URL`."""
        return self.get("OLLAMA_BASE_URL") or self.OLLAMA_HOST

    # Centroids
    @property
    def CENTROIDS_DIM(self) -> int:
        """Obtém o valor da configuração `CENTROIDS_DIM`."""
        return int(self.get("CENTROIDS_DIM"))
    @property
    def CENTROIDS_K(self) -> int:
        """Obtém o valor da configuração `CENTROIDS_K`."""
        return int(self.get("CENTROIDS_K"))
    @property
    def CENTROIDS_MIN_SIM_CREATE(self) -> float:
        """Obtém o valor da configuração `CENTROIDS_MIN_SIM_CREATE`."""
        return float(self.get("CENTROIDS_MIN_SIM_CREATE"))
    @property
    def CENTROIDS_ENABLE_ONLINE(self) -> bool:
        """Obtém o valor da configuração `CENTROIDS_ENABLE_ONLINE`."""
        return self._get_bool("CENTROIDS_ENABLE_ONLINE", True)
    @property
    def CENTROIDS_UPDATE_INTERVAL_S(self) -> int:
        """Obtém o valor da configuração `CENTROIDS_UPDATE_INTERVAL_S`."""
        return int(self.get("CENTROIDS_UPDATE_INTERVAL_S"))
    @property
    def CENTROIDS_MIN_RECORDS_FOR_TRAIN(self) -> int:
        """Obtém o valor da configuração `CENTROIDS_MIN_RECORDS_FOR_TRAIN`."""
        return int(self.get("CENTROIDS_MIN_RECORDS_FOR_TRAIN"))
    @property
    def CENTROIDS_MAX_HISTORY(self) -> int:
        """Obtém o valor da configuração `CENTROIDS_MAX_HISTORY`."""
        return int(self.get("CENTROIDS_MAX_HISTORY"))
    @property
    def CENTROIDS_HOURLY_REFRESH_ENABLED(self) -> bool:
        """Obtém o valor da configuração `CENTROIDS_HOURLY_REFRESH_ENABLED`."""
        return str(self.get("CENTROIDS_HOURLY_REFRESH_ENABLED")).strip() in ("1", "true", "True")
    @property
    def CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH(self) -> int:
        """Obtém o valor da configuração `CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH`."""
        return int(self.get("CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH"))

    # NSGA / Meta
    @property
    def NSGA_UPDATE_INTERVAL_S(self) -> int:
        """Obtém o valor da configuração `NSGA_UPDATE_INTERVAL_S`."""
        return int(self.get("NSGA_UPDATE_INTERVAL_S", "300"))
    @property
    def NSGA_LOOKBACK_MINUTES(self) -> int:
        """Obtém o valor da configuração `NSGA_LOOKBACK_MINUTES`."""
        return int(self.get("NSGA_LOOKBACK_MINUTES", "180"))
    @property
    def NSGA_LOOKBACK_MAXROWS(self) -> int:
        """Obtém o valor da configuração `NSGA_LOOKBACK_MAXROWS`."""
        return int(self.get("NSGA_LOOKBACK_MAXROWS", "2000"))
    @property
    def METAOPT_REPS(self) -> int:
        """Obtém o valor da configuração `METAOPT_REPS`."""
        return int(self.get("METAOPT_REPS", "5"))
    @property
    def METAOPT_TRIALS(self) -> int:
        """Obtém o valor da configuração `METAOPT_TRIALS`."""
        return int(self.get("METAOPT_TRIALS", "100"))

    # Propriedades de Pesos NSGA-II
    @property
    def NSGA_W_QUALITY(self) -> float:
        """Obtém o valor da configuração `NSGA_W_QUALITY`."""
        return float(self.get("NSGA_W_QUALITY", 1.0))
    @property
    def NSGA_W_LATENCY(self) -> float:
        """Obtém o valor da configuração `NSGA_W_LATENCY`."""
        return float(self.get("NSGA_W_LATENCY", 0.5))
    @property
    def NSGA_W_COST(self) -> float:
        """Obtém o valor da configuração `NSGA_W_COST`."""
        return float(self.get("NSGA_W_COST", 50.0))
    @property
    def NSGA_W_ALIGNMENT(self) -> float:
        """Obtém o valor da configuração `NSGA_W_ALIGNMENT`."""
        return float(self.get("NSGA_W_ALIGNMENT", 1.0))

    # Phase 1: Monitoring Properties
    @property
    def NSGA_CONVERGENCE_HISTORY_SIZE(self) -> int:
        """Obtém o valor da configuração `NSGA_CONVERGENCE_HISTORY_SIZE`."""
        return int(self.get("NSGA_CONVERGENCE_HISTORY_SIZE", 20))
    @property
    def CASCADE_WARNING_THRESHOLD(self) -> float:
        """Obtém o valor da configuração `CASCADE_WARNING_THRESHOLD`."""
        return float(self.get("CASCADE_WARNING_THRESHOLD", 0.3))
    @property
    def CASCADE_CRITICAL_THRESHOLD(self) -> float:
        """Obtém o valor da configuração `CASCADE_CRITICAL_THRESHOLD`."""
        return float(self.get("CASCADE_CRITICAL_THRESHOLD", 0.5))

    # Phase 2: Self-Tuning Properties
    @property
    def RISK_FACTOR_SOTA_HIGH_UQ(self) -> float:
        """Obtém o valor da configuração `RISK_FACTOR_SOTA_HIGH_UQ`."""
        return float(self.get("RISK_FACTOR_SOTA_HIGH_UQ", 1.3))
    @property
    def RISK_FACTOR_LOCAL_HIGH_UQ(self) -> float:
        """Obtém o valor da configuração `RISK_FACTOR_LOCAL_HIGH_UQ`."""
        return float(self.get("RISK_FACTOR_LOCAL_HIGH_UQ", 0.6))
    @property
    def RISK_FACTOR_LOCAL_LOW_UQ(self) -> float:
        """Obtém o valor da configuração `RISK_FACTOR_LOCAL_LOW_UQ`."""
        return float(self.get("RISK_FACTOR_LOCAL_LOW_UQ", 1.1))
    @property
    def RISK_FACTOR_ADAPT_ENABLED(self) -> bool:
        """Obtém o valor da configuração `RISK_FACTOR_ADAPT_ENABLED`."""
        return self._get_bool("RISK_FACTOR_ADAPT_ENABLED", False)
    @property
    def RISK_FACTOR_ADAPT_RATE(self) -> float:
        """Obtém o valor da configuração `RISK_FACTOR_ADAPT_RATE`."""
        return float(self.get("RISK_FACTOR_ADAPT_RATE", 0.02))
    @property
    def ADAPTIVE_TIMEOUT_ENABLED(self) -> bool:
        """Obtém o valor da configuração `ADAPTIVE_TIMEOUT_ENABLED`."""
        return self._get_bool("ADAPTIVE_TIMEOUT_ENABLED", False)
    @property
    def ADAPTIVE_TIMEOUT_MULTIPLIER(self) -> float:
        """Obtém o valor da configuração `ADAPTIVE_TIMEOUT_MULTIPLIER`."""
        return float(self.get("ADAPTIVE_TIMEOUT_MULTIPLIER", 2.0))
    @property
    def ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER(self) -> float:
        """Obtém o valor da configuração `ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER`."""
        return float(self.get("ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER", 3.0))
    @property
    def MIN_TIMEOUT(self) -> int:
        """Obtém o valor da configuração `MIN_TIMEOUT`."""
        return int(self.get("MIN_TIMEOUT", 30))
    @property
    def MAX_TIMEOUT(self) -> int:
        """Obtém o valor da configuração `MAX_TIMEOUT`."""
        return int(self.get("MAX_TIMEOUT", 1200))
    @property
    def META_OPT_ENABLED(self) -> bool:
        """Obtém o valor da configuração `META_OPT_ENABLED`."""
        return self._get_bool("META_OPT_ENABLED", False)
    @property
    def META_OPT_SCHEDULE_HOUR(self) -> int:
        """Obtém o valor da configuração `META_OPT_SCHEDULE_HOUR`."""
        return int(self.get("META_OPT_SCHEDULE_HOUR", 3))
    @property
    def META_OPT_SCHEDULED_TRIALS(self) -> int:
        """Obtém o valor da configuração `META_OPT_SCHEDULED_TRIALS`."""
        return int(self.get("META_OPT_SCHEDULED_TRIALS", 20))

    # Phase 3: Feedback Properties
    @property
    def DRIFT_THRESHOLD(self) -> float:
        """Obtém o valor da configuração `DRIFT_THRESHOLD`."""
        return float(self.get("DRIFT_THRESHOLD", 0.15))
    @property
    def DRIFT_WINDOW_SIZE(self) -> int:
        """Obtém o valor da configuração `DRIFT_WINDOW_SIZE`."""
        return int(self.get("DRIFT_WINDOW_SIZE", 100))
    @property
    def USER_FEEDBACK_WEIGHT(self) -> float:
        """Obtém o valor da configuração `USER_FEEDBACK_WEIGHT`."""
        return float(self.get("USER_FEEDBACK_WEIGHT", 0.7))

    # Phase 4: A/B Testing Properties
    @property
    def AB_TESTING_ENABLED(self) -> bool:
        """Obtém o valor da configuração `AB_TESTING_ENABLED`."""
        return self._get_bool("AB_TESTING_ENABLED", False)

    # Phase 5: Autonomous Behavior - Adaptive Cache Properties
    @property
    def CACHE_THRESHOLD_MIN(self) -> float:
        """Obtém o valor da configuração `CACHE_THRESHOLD_MIN`."""
        return float(self.get("CACHE_THRESHOLD_MIN", 0.85))
    @property
    def CACHE_THRESHOLD_MAX(self) -> float:
        """Obtém o valor da configuração `CACHE_THRESHOLD_MAX`."""
        return float(self.get("CACHE_THRESHOLD_MAX", 0.98))
    @property
    def CACHE_HIT_RATE_TARGET(self) -> float:
        """Obtém o valor da configuração `CACHE_HIT_RATE_TARGET`."""
        return float(self.get("CACHE_HIT_RATE_TARGET", 0.20))
    @property
    def CACHE_THRESHOLD_ADAPT_ENABLED(self) -> bool:
        """Obtém o valor da configuração `CACHE_THRESHOLD_ADAPT_ENABLED`."""
        return self._get_bool("CACHE_THRESHOLD_ADAPT_ENABLED", False)

    # Phase 5: Autonomous Behavior - Predictor Validation Properties
    @property
    def PREDICTOR_VALIDATION_ENABLED(self) -> bool:
        """Obtém o valor da configuração `PREDICTOR_VALIDATION_ENABLED`."""
        return self._get_bool("PREDICTOR_VALIDATION_ENABLED", True)
    @property
    def PREDICTOR_BRIER_SCORE_THRESHOLD(self) -> float:
        """Obtém o valor da configuração `PREDICTOR_BRIER_SCORE_THRESHOLD`."""
        return float(self.get("PREDICTOR_BRIER_SCORE_THRESHOLD", 0.25))
    @property
    def PREDICTOR_CALIBRATION_WINDOW(self) -> int:
        """Obtém o valor da configuração `PREDICTOR_CALIBRATION_WINDOW`."""
        return int(self.get("PREDICTOR_CALIBRATION_WINDOW", 1000))

    # Phase 5: Autonomous Behavior - UQ Calibration Properties
    @property
    def UQ_CALIBRATION_ENABLED(self) -> bool:
        """Obtém o valor da configuração `UQ_CALIBRATION_ENABLED`."""
        return self._get_bool("UQ_CALIBRATION_ENABLED", True)
    @property
    def UQ_QUALITY_GAP_RELAX(self) -> float:
        """Obtém o valor da configuração `UQ_QUALITY_GAP_RELAX`."""
        return float(self.get("UQ_QUALITY_GAP_RELAX", 0.5))
    @property
    def UQ_QUALITY_GAP_TIGHTEN(self) -> float:
        """Obtém o valor da configuração `UQ_QUALITY_GAP_TIGHTEN`."""
        return float(self.get("UQ_QUALITY_GAP_TIGHTEN", 2.0))

    # Phase 5: Autonomous Behavior - Judge Calibration Properties
    @property
    def JUDGE_CALIBRATION_ENABLED(self) -> bool:
        """Obtém o valor da configuração `JUDGE_CALIBRATION_ENABLED`."""
        return self._get_bool("JUDGE_CALIBRATION_ENABLED", True)
    @property
    def JUDGE_CACHE_AGREEMENT_TARGET(self) -> float:
        """Obtém o valor da configuração `JUDGE_CACHE_AGREEMENT_TARGET`."""
        return float(self.get("JUDGE_CACHE_AGREEMENT_TARGET", 0.7))

    # Circuit Breaker Properties
    @property
    def CIRCUIT_BREAKER_FAIL_MAX(self) -> int:
        """Obtém o valor da configuração `CIRCUIT_BREAKER_FAIL_MAX`."""
        return int(self.get("CIRCUIT_BREAKER_FAIL_MAX", 5))
    @property
    def CIRCUIT_BREAKER_RESET_TIMEOUT(self) -> int:
        """Obtém o valor da configuração `CIRCUIT_BREAKER_RESET_TIMEOUT`."""
        return int(self.get("CIRCUIT_BREAKER_RESET_TIMEOUT", 60))
    @property
    def CIRCUIT_BREAKER_LOCAL_FAIL_MAX(self) -> int:
        """Obtém o valor da configuração `CIRCUIT_BREAKER_LOCAL_FAIL_MAX`."""
        return int(self.get("CIRCUIT_BREAKER_LOCAL_FAIL_MAX", 3))
    @property
    def CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT(self) -> int:
        """Obtém o valor da configuração `CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT`."""
        return int(self.get("CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT", 30))

    # Backpressure / Concurrency Control
    @property
    def MAX_CONCURRENT_REQUESTS(self) -> int:
        """Obtém o valor da configuração `MAX_CONCURRENT_REQUESTS`."""
        return int(self.get("MAX_CONCURRENT_REQUESTS", 500))
    @property
    def BACKPRESSURE_ENABLED(self) -> bool:
        """Obtém o valor da configuração `BACKPRESSURE_ENABLED`."""
        return self._get_bool("BACKPRESSURE_ENABLED", True)

    # Emergency Fallback Models
    @property
    def EMERGENCY_FALLBACK_MODELS(self) -> List[str]:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return self._get_list("EMERGENCY_FALLBACK_MODELS")


settings = DynamicSettings()


# ============================================================
# DB Connection Pool Metrics (Quick Win #10)
# ============================================================

def get_db_pool_stats() -> dict:
    """Get database connection pool statistics."""
    try:
        pool = engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalidatedcount() if hasattr(pool, 'invalidatedcount') else 0,
        }
    except Exception:
        return {"size": 0, "checked_in": 0, "checked_out": 0, "overflow": 0, "invalid": 0}


def update_db_pool_metrics():
    """Update Prometheus metrics for DB pool (call periodically)."""
    try:
        from app.observability import (
            DB_POOL_SIZE,
            DB_POOL_CHECKED_IN,
            DB_POOL_CHECKED_OUT,
            DB_POOL_OVERFLOW,
        )
        stats = get_db_pool_stats()
        DB_POOL_SIZE.set(stats["size"])
        DB_POOL_CHECKED_IN.set(stats["checked_in"])
        DB_POOL_CHECKED_OUT.set(stats["checked_out"])
        DB_POOL_OVERFLOW.set(stats["overflow"])
    except Exception:
        pass  # Metrics not available


def validate_critical_settings(settings_obj: Optional[Any] = None) -> List[str]:
    """
    Validate critical runtime settings.
    Returns a list of validation errors (empty when valid).
    """
    errors: List[str] = []
    cfg = settings_obj or settings

    def _read(name: str, default: Any) -> Any:
        """Executa a responsabilidade descrita por este método.

        Args:
            name: Parâmetro de entrada.
            default: Parâmetro de entrada.

        Returns:
            Valor produzido pela execução.
        """
        try:
            if hasattr(cfg, name):
                return getattr(cfg, name)
        except Exception:
            pass
        try:
            getter = getattr(cfg, "get", None)
            if callable(getter):
                return getter(name, default)
        except Exception:
            pass
        return default

    try:
        min_timeout = as_int(_read("MIN_TIMEOUT", 30), 30)
        max_timeout = as_int(_read("MAX_TIMEOUT", 1200), 1200)
        if min_timeout <= 0:
            errors.append("MIN_TIMEOUT must be > 0")
        if max_timeout <= 0:
            errors.append("MAX_TIMEOUT must be > 0")
        if min_timeout > max_timeout:
            errors.append("MIN_TIMEOUT must be <= MAX_TIMEOUT")
    except Exception:
        errors.append("Timeout settings are invalid")

    try:
        cq = as_float(_read("NSGA_W_QUALITY", 1.0), 1.0)
        cl = as_float(_read("NSGA_W_LATENCY", 0.5), 0.5)
        cc = as_float(_read("NSGA_W_COST", 100.0), 100.0)
        if cq < 0 or cl < 0 or cc < 0:
            errors.append("NSGA weights must be non-negative")
        if (cq + cl + cc) <= 0:
            errors.append("NSGA weights sum must be > 0")
    except Exception:
        errors.append("NSGA weights are invalid")

    return errors


# ============================================================
# Hot-reload listener via Redis Pub/Sub (COM RECONEXÃO)
# ============================================================

_reload_listener_thread: Optional[threading.Thread] = None
_reload_listener_stop = threading.Event()


def start_reload_listener() -> None:
    """Executa a responsabilidade descrita por este método.

    Returns:
        Valor produzido pela execução.
    """
    global _reload_listener_thread
    if _reload_listener_thread and _reload_listener_thread.is_alive():
        return
    _reload_listener_stop.clear()

    # Usamos uma conexão dedicada para o listener, sem timeout
    def _bg():
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        while not _reload_listener_stop.is_set():
            listener = None
            pubsub = None
            try:
                # Conexão dedicada para o PubSub com timeout infinito
                # Importa redis aqui para evitar dependência circular se fosse no topo
                import redis
                
                listener = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD or None,
                    socket_timeout=None, # <--- O SEGREDO
                    socket_keepalive=True
                )
                
                pubsub = listener.pubsub()
                pubsub.subscribe(REDIS_RELOAD_CHANNEL)
                logger.info(f"[settings_dynamic] Listener hot-reload conectado: {REDIS_RELOAD_CHANNEL}")
                
                while not _reload_listener_stop.is_set():
                    msg = pubsub.get_message(timeout=1.0)
                    if msg and msg.get("type") == "message":
                        key = msg.get("data")
                        if isinstance(key, bytes):
                            key = key.decode()
                        logger.info(f"[settings_dynamic] Hot-reload: '{key}'.")
                        _invalidate_cache()
            
            except Exception as e:
                logger.error(f"[settings_dynamic] Erro no listener: {e}. Reconectando em 5s...")
                _reload_listener_stop.wait(5)
            finally:
                try:
                    if pubsub:
                        pubsub.close()
                except Exception:
                    pass
                try:
                    if listener:
                        listener.close()
                except Exception:
                    pass

    _reload_listener_thread = threading.Thread(target=_bg, daemon=True, name="settings-reload-listener")
    _reload_listener_thread.start()


def stop_reload_listener() -> None:
    """Executa a responsabilidade descrita por este método.

    Returns:
        Valor produzido pela execução.
    """
    global _reload_listener_thread
    _reload_listener_stop.set()
    if _reload_listener_thread and _reload_listener_thread.is_alive():
        _reload_listener_thread.join(timeout=1.0)
    _reload_listener_thread = None
