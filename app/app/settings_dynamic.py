# -*- coding: utf-8 -*-
"""
settings_dynamic.py (VERSÃO FINAL: Com Configuração de Amostragem)
--------------------------------------------------------------------------
Carrega configurações com fallback em camadas.
"""

from __future__ import annotations

import os
import json
import time
import logging
import threading
import redis
from collections import OrderedDict
from typing import Any, Dict, Optional, List

from sqlalchemy import text
from app.utils.redis_client import get_redis_async_safe, ensure_redis_connected

logger = logging.getLogger(__name__)

# ============================================================
# Redis / Banco básicos
# ============================================================

REDIS_PREFIX = "settings:"
REDIS_RELOAD_CHANNEL = "settings:reload"

def _get_rds():
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
        return _get_settings_engine().begin()

    def connect(self):
        return _get_settings_engine().connect()

    @property
    def pool(self):
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
    def __init__(self, maxsize: int = 512, ttl_s: int = 30):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
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
        now = time.time()
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (value, now)
            if len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


SETTINGS_CACHE_SIZE = int(os.getenv("SETTINGS_CACHE_SIZE", "2000"))  # Optimized for high-capacity
SETTINGS_CACHE_TTL_S = int(os.getenv("SETTINGS_CACHE_TTL_S", "300"))  # 5 min - reduces Redis/DB lookups by ~80%

_lru = LRUCache(maxsize=SETTINGS_CACHE_SIZE, ttl_s=SETTINGS_CACHE_TTL_S)


def _invalidate_cache():
    _lru.clear()
    logger.info("[settings_dynamic] Cache LRU invalidado.")


# ============================================================
# Auxiliares de acesso Redis / DB
# ============================================================

def _get_from_redis(key: str) -> Optional[str]:
    rds = _get_rds()
    if not rds:
        return None
    try:
        v = rds.get(f"{REDIS_PREFIX}{key}")
        if v is None:
            return None
        if isinstance(v, bytes):
            return v.decode()
        if isinstance(v, (str, int, float, bool)):
            return str(v)
        # Ignore mocked/non-serializable Redis values in tests.
        return None
    except Exception:
        pass
    return None


def _get_from_db(key: str) -> Optional[str]:
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
    rds = _get_rds()
    if not rds:
        return
    try:
        rds.set(f"{REDIS_PREFIX}{key}", val)
    except Exception as e:
        logger.debug(f"[settings_dynamic] Falha ao gravar Redis para {key}: {e}")


def _load_json_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return [x.strip() for x in raw.split(",") if x.strip()]


# ============================================================
# Classe Principal (Integrada)
# ============================================================

class DynamicSettings:

    DEFAULTS: Dict[str, str] = {
        "MAX_TOKENS_DEFAULT": "2000",
        "TEMPERATURE_DEFAULT": "0.55",
        "BANDIT_EPSILON": "0.12",
        "QUERY_LOG_RETENTION_DAYS": "7",

        # Redis defaults
        "REDIS_HOST": "redis",
        "REDIS_PORT": "6379",
        "REDIS_DB": "0",
        "REDIS_PASSWORD": "",

        # Embeddings base
        "EMBED_MODEL": "nomic-embed-text",
        "EMBED_PROVIDER": "ollama",
        "EMBED_DEVICE": "cpu",
        "EMBED_TEXT_MODEL": "nomic-embed-text",

        # Embeddings multimodais e visão
        "TEXT_EMBEDDING_MODEL": "nomic-embed-text",
        "IMAGE_EMBEDDING_MODEL": "clip-vit-large-patch14",
        "MULTIMODAL_EMBEDDING_MODEL": "gpt-4o-mini-embed",

        # Centróides (router/bandit)
        "CENTROIDS_DIM": "768",
        "CENTROIDS_K": "20",
        "CENTROIDS_MIN_SIM_CREATE": "0.35",
        "CENTROIDS_ENABLE_ONLINE": "1",
        "CENTROIDS_UPDATE_INTERVAL_S": "1800",
        "CENTROIDS_MIN_RECORDS_FOR_TRAIN": "50",
        "CENTROIDS_MAX_HISTORY": "50000",
        "CENTROIDS_HOURLY_REFRESH_ENABLED": "1",
        "CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH": "50",

        # Judges
        "JUDGES_ENABLED": "1",
        "JUDGES_MODE": "llm",
        "JUDGES_LOCAL_MODEL": "ollama/phi4:latest",
        "JUDGES_REMOTE_MODEL": "gpt-5-mini",
        "JUDGES_TIMEOUT_S": "15",
        "JUDGE_MIN_SAMPLE_RATE": "0.05", # <--- NOVO: Mínimo 5% de amostragem

        # Ollama
        "OLLAMA_BASE_URL": "http://ollama:11434",
        "OLLAMA_HOST": "http://ollama:11434",
        "OLLAMA_CONCURRENCY_LIMIT": "30",  # Quick Win #7: Configurable concurrency

        # Listas multimodais
        "CANDIDATE_MODELS_LIST": "[]",
        "CANDIDATE_VISION_MODELS_LIST": "[]",
        "CANDIDATE_MULTIMODAL_MODELS_LIST": "[]",
        
        # Modelos VLM padrão
        "VLM_OLLAMA_MODELS": json.dumps([
            "qwen3-vl:8b",
            "gemma3:4b",
            "llama3.2:3b",
            "llama3:8b",
            "llava:7b",
            "llama3.2-vision:11b",
            "llava-llama3:8b",
            "granite3.2-vision:2b",
        ]),
        
        # Cache & RAG
        "CACHE_TTL_DAYS": "7",
        "CACHE_THRESHOLD": "0.92",
        "UNCERTAINTY_THRESHOLD": "0.7",
        "RERANK_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "RERANK_ENABLED": "1",
        "RAG_DATA_DIR": "/app/data",

        # Pesos NSGA-II Dinâmicos
        "NSGA_W_QUALITY": "1.0",
        "NSGA_W_LATENCY": "0.5",
        "NSGA_W_COST": "100.0",
        "NSGA_W_ALIGNMENT": "1.0",

        # Phase 1: Monitoring
        "NSGA_CONVERGENCE_HISTORY_SIZE": "20",
        "CASCADE_WARNING_THRESHOLD": "0.3",
        "CASCADE_CRITICAL_THRESHOLD": "0.5",

        # Phase 2: Self-Tuning
        "RISK_FACTOR_SOTA_HIGH_UQ": "1.3",
        "RISK_FACTOR_LOCAL_HIGH_UQ": "0.6",
        "RISK_FACTOR_LOCAL_LOW_UQ": "1.1",
        "RISK_FACTOR_ADAPT_ENABLED": "0",
        "RISK_FACTOR_ADAPT_RATE": "0.02",
        "ADAPTIVE_TIMEOUT_ENABLED": "0",
        "ADAPTIVE_TIMEOUT_MULTIPLIER": "2.0",
        "ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER": "3.0",
        "MIN_TIMEOUT": "30",
        "MAX_TIMEOUT": "1200",
        "META_OPT_ENABLED": "0",
        "META_OPT_SCHEDULE_HOUR": "3",
        "META_OPT_SCHEDULED_TRIALS": "20",

        # Phase 3: Feedback
        "DRIFT_THRESHOLD": "0.15",
        "DRIFT_WINDOW_SIZE": "100",
        "USER_FEEDBACK_WEIGHT": "0.7",

        # Phase 4: A/B Testing
        "AB_TESTING_ENABLED": "0",

        # Request-level settings
        "REQUEST_TIMEOUT_SECONDS": "120",
        "REQUEST_DEDUP_ENABLED": "1",

        # Circuit breaker settings
        "CIRCUIT_BREAKER_FAIL_MAX": "5",
        "CIRCUIT_BREAKER_RESET_TIMEOUT": "60",
        "CIRCUIT_BREAKER_LOCAL_FAIL_MAX": "3",
        "CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT": "30",

        # Backpressure / concurrency control
        "MAX_CONCURRENT_REQUESTS": "500",
        "BACKPRESSURE_ENABLED": "1",

        # Emergency fallback models (used when cascade failure detected)
        "EMERGENCY_FALLBACK_MODELS": json.dumps([
            "ollama/phi4:latest",
            "ollama/gemma3:4b",
            "ollama/llama3:8b",
        ]),

        # Phase 5: Autonomous Behavior - Adaptive Cache Threshold
        "CACHE_THRESHOLD_MIN": "0.85",
        "CACHE_THRESHOLD_MAX": "0.98",
        "CACHE_HIT_RATE_TARGET": "0.20",
        "CACHE_THRESHOLD_ADAPT_ENABLED": "0",

        # Phase 5: Autonomous Behavior - Predictor Validation
        "PREDICTOR_VALIDATION_ENABLED": "1",
        "PREDICTOR_BRIER_SCORE_THRESHOLD": "0.25",
        "PREDICTOR_CALIBRATION_WINDOW": "1000",

        # Phase 5: Autonomous Behavior - UQ Calibration
        "UQ_CALIBRATION_ENABLED": "1",
        "UQ_QUALITY_GAP_RELAX": "0.5",
        "UQ_QUALITY_GAP_TIGHTEN": "2.0",

        # Phase 5: Autonomous Behavior - Judge Calibration
        "JUDGE_CALIBRATION_ENABLED": "1",
        "JUDGE_CACHE_AGREEMENT_TARGET": "0.7",
    }

    def get(self, key: str, fallback: Any = None) -> Any:
        cached = _lru.get(key)
        if cached is not None:
            return cached
        v = _get_from_redis(key)
        if v is None:
            v = _get_from_db(key)
        if v is None:
            v = os.getenv(key)
        if v is None:
            v = self.DEFAULTS.get(key, fallback)
        if v is not None:
            _lru.set(key, v)
        return v

    def set(self, key: str, value: str, actor: str = "system", source: str = "internal") -> None:
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
        keys = list(self.DEFAULTS.keys()) + [
            "DB_HOST", "DB_USER", "DB_NAME", "DB_PORT", 
            "OLLAMA_HOST", "OLLAMA_BASE_URL"
        ]
        out = {}
        for k in keys:
            try:
                out[k] = self.get(k)
            except Exception:
                out[k] = None
        return out

    # -------------------------
    # Propriedades Tipadas
    # -------------------------
    @property
    def CANDIDATE_MODELS_LIST(self) -> List[str]:
        return _load_json_list(self.get("CANDIDATE_MODELS_LIST", "[]"))

    @property
    def CANDIDATE_VISION_MODELS_LIST(self) -> List[str]:
        return _load_json_list(self.get("CANDIDATE_VISION_MODELS_LIST", "[]"))

    @property
    def CANDIDATE_MULTIMODAL_MODELS_LIST(self) -> List[str]:
        return _load_json_list(self.get("CANDIDATE_MULTIMODAL_MODELS_LIST", "[]"))

    @property
    def VLM_OLLAMA_MODELS(self) -> List[str]:
        return _load_json_list(self.get("VLM_OLLAMA_MODELS", "[]"))
    
    @property
    def JUDGE_MODELS(self) -> List[str]:
        return _load_json_list(self.get("JUDGE_MODELS", "[]"))

    # Embeddings
    @property
    def EMBED_TEXT_MODEL(self) -> str:
        return self.get("EMBED_TEXT_MODEL", "nomic-embed-text")

    @property
    def TEXT_EMBEDDING_MODEL(self) -> str: return self.get("TEXT_EMBEDDING_MODEL")
    @property
    def IMAGE_EMBEDDING_MODEL(self) -> str: return self.get("IMAGE_EMBEDDING_MODEL")
    @property
    def MULTIMODAL_EMBEDDING_MODEL(self) -> str: return self.get("MULTIMODAL_EMBEDDING_MODEL")
    @property
    def EMBED_MODEL(self) -> str: return self.get("EMBED_MODEL")
    @property
    def EMBED_PROVIDER(self) -> str: return self.get("EMBED_PROVIDER")
    @property
    def EMBED_DEVICE(self) -> str: return self.get("EMBED_DEVICE")

    # Router Params
    @property
    def MAX_TOKENS_DEFAULT(self) -> int: return int(self.get("MAX_TOKENS_DEFAULT", 2000))
    @property
    def TEMPERATURE_DEFAULT(self) -> float: return float(self.get("TEMPERATURE_DEFAULT", 0.5))
    @property
    def BANDIT_EPSILON(self) -> float: return float(self.get("BANDIT_EPSILON"))
    @property
    def QUERY_LOG_RETENTION_DAYS(self) -> int: return int(self.get("QUERY_LOG_RETENTION_DAYS"))
    
    # Banco / Infra
    @property
    def REDIS_HOST(self) -> str: return self.get("REDIS_HOST")
    @property
    def REDIS_PORT(self) -> int: return int(self.get("REDIS_PORT"))
    @property
    def REDIS_DB(self) -> int: return int(self.get("REDIS_DB"))
    @property
    def REDIS_PASSWORD(self) -> str: return self.get("REDIS_PASSWORD")

    @property
    def DB_HOST(self) -> str: return self.get("DB_HOST", DB_HOST_ENV)
    @property
    def DB_PORT(self) -> int: return int(self.get("DB_PORT", str(DB_PORT_ENV)))
    @property
    def DB_USER(self) -> str: return self.get("DB_USER", DB_USER_ENV)
    @property
    def DB_PASS(self) -> str: return self.get("DB_PASS", DB_PASS_ENV)
    @property
    def DB_NAME(self) -> str: return self.get("DB_NAME", DB_NAME_ENV)
    
    @property
    def ADMIN_TOKEN(self) -> str: return self.get("ADMIN_TOKEN", "")
    @property
    def ADMIN_TOKEN_PREVIOUS(self) -> str: return self.get("ADMIN_TOKEN_PREVIOUS", "")

    # Judges
    @property
    def JUDGES_ENABLED(self) -> bool: return str(self.get("JUDGES_ENABLED")).strip() in ("1", "true", "True")
    @property
    def JUDGES_MODE(self) -> str: return self.get("JUDGES_MODE")
    @property
    def JUDGES_LOCAL_MODEL(self) -> str: return self.get("JUDGES_LOCAL_MODEL")
    @property
    def JUDGES_REMOTE_MODEL(self) -> str: return self.get("JUDGES_REMOTE_MODEL")
    @property
    def JUDGES_TIMEOUT_S(self) -> int: return int(self.get("JUDGES_TIMEOUT_S"))
    @property
    def JUDGE_MIN_SAMPLE_RATE(self) -> float: return float(self.get("JUDGE_MIN_SAMPLE_RATE", 0.05))

    # Ollama
    @property
    def OLLAMA_HOST(self) -> str: return self.get("OLLAMA_HOST") or "http://ollama:11434"
    @property
    def OLLAMA_BASE_URL(self) -> str: return self.get("OLLAMA_BASE_URL") or self.OLLAMA_HOST

    # Centroids
    @property
    def CENTROIDS_DIM(self) -> int: return int(self.get("CENTROIDS_DIM"))
    @property
    def CENTROIDS_K(self) -> int: return int(self.get("CENTROIDS_K"))
    @property
    def CENTROIDS_MIN_SIM_CREATE(self) -> float: return float(self.get("CENTROIDS_MIN_SIM_CREATE"))
    @property
    def CENTROIDS_ENABLE_ONLINE(self) -> bool: return str(self.get("CENTROIDS_ENABLE_ONLINE")).strip() in ("1", "true", "True")
    @property
    def CENTROIDS_UPDATE_INTERVAL_S(self) -> int: return int(self.get("CENTROIDS_UPDATE_INTERVAL_S"))
    @property
    def CENTROIDS_MIN_RECORDS_FOR_TRAIN(self) -> int: return int(self.get("CENTROIDS_MIN_RECORDS_FOR_TRAIN"))
    @property
    def CENTROIDS_MAX_HISTORY(self) -> int: return int(self.get("CENTROIDS_MAX_HISTORY"))
    @property
    def CENTROIDS_HOURLY_REFRESH_ENABLED(self) -> bool: return str(self.get("CENTROIDS_HOURLY_REFRESH_ENABLED")).strip() in ("1", "true", "True")
    @property
    def CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH(self) -> int: return int(self.get("CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH"))

    # NSGA / Meta
    @property
    def NSGA_UPDATE_INTERVAL_S(self) -> int: return int(self.get("NSGA_UPDATE_INTERVAL_S", "300"))
    @property
    def NSGA_LOOKBACK_MINUTES(self) -> int: return int(self.get("NSGA_LOOKBACK_MINUTES", "180"))
    @property
    def NSGA_LOOKBACK_MAXROWS(self) -> int: return int(self.get("NSGA_LOOKBACK_MAXROWS", "2000"))
    @property
    def METAOPT_REPS(self) -> int: return int(self.get("METAOPT_REPS", "5"))
    @property
    def METAOPT_TRIALS(self) -> int: return int(self.get("METAOPT_TRIALS", "100"))

    # Propriedades de Pesos NSGA-II
    @property
    def NSGA_W_QUALITY(self) -> float: return float(self.get("NSGA_W_QUALITY", 1.0))
    @property
    def NSGA_W_LATENCY(self) -> float: return float(self.get("NSGA_W_LATENCY", 0.5))
    @property
    def NSGA_W_COST(self) -> float: return float(self.get("NSGA_W_COST", 50.0))
    @property
    def NSGA_W_ALIGNMENT(self) -> float: return float(self.get("NSGA_W_ALIGNMENT", 1.0))

    # Phase 1: Monitoring Properties
    @property
    def NSGA_CONVERGENCE_HISTORY_SIZE(self) -> int: return int(self.get("NSGA_CONVERGENCE_HISTORY_SIZE", 20))
    @property
    def CASCADE_WARNING_THRESHOLD(self) -> float: return float(self.get("CASCADE_WARNING_THRESHOLD", 0.3))
    @property
    def CASCADE_CRITICAL_THRESHOLD(self) -> float: return float(self.get("CASCADE_CRITICAL_THRESHOLD", 0.5))

    # Phase 2: Self-Tuning Properties
    @property
    def RISK_FACTOR_SOTA_HIGH_UQ(self) -> float: return float(self.get("RISK_FACTOR_SOTA_HIGH_UQ", 1.3))
    @property
    def RISK_FACTOR_LOCAL_HIGH_UQ(self) -> float: return float(self.get("RISK_FACTOR_LOCAL_HIGH_UQ", 0.6))
    @property
    def RISK_FACTOR_LOCAL_LOW_UQ(self) -> float: return float(self.get("RISK_FACTOR_LOCAL_LOW_UQ", 1.1))
    @property
    def RISK_FACTOR_ADAPT_ENABLED(self) -> bool: return str(self.get("RISK_FACTOR_ADAPT_ENABLED")).strip() in ("1", "true", "True")
    @property
    def RISK_FACTOR_ADAPT_RATE(self) -> float: return float(self.get("RISK_FACTOR_ADAPT_RATE", 0.02))
    @property
    def ADAPTIVE_TIMEOUT_ENABLED(self) -> bool: return str(self.get("ADAPTIVE_TIMEOUT_ENABLED")).strip() in ("1", "true", "True")
    @property
    def ADAPTIVE_TIMEOUT_MULTIPLIER(self) -> float: return float(self.get("ADAPTIVE_TIMEOUT_MULTIPLIER", 2.0))
    @property
    def ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER(self) -> float: return float(self.get("ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER", 3.0))
    @property
    def MIN_TIMEOUT(self) -> int: return int(self.get("MIN_TIMEOUT", 30))
    @property
    def MAX_TIMEOUT(self) -> int: return int(self.get("MAX_TIMEOUT", 1200))
    @property
    def META_OPT_ENABLED(self) -> bool: return str(self.get("META_OPT_ENABLED")).strip() in ("1", "true", "True")
    @property
    def META_OPT_SCHEDULE_HOUR(self) -> int: return int(self.get("META_OPT_SCHEDULE_HOUR", 3))
    @property
    def META_OPT_SCHEDULED_TRIALS(self) -> int: return int(self.get("META_OPT_SCHEDULED_TRIALS", 20))

    # Phase 3: Feedback Properties
    @property
    def DRIFT_THRESHOLD(self) -> float: return float(self.get("DRIFT_THRESHOLD", 0.15))
    @property
    def DRIFT_WINDOW_SIZE(self) -> int: return int(self.get("DRIFT_WINDOW_SIZE", 100))
    @property
    def USER_FEEDBACK_WEIGHT(self) -> float: return float(self.get("USER_FEEDBACK_WEIGHT", 0.7))

    # Phase 4: A/B Testing Properties
    @property
    def AB_TESTING_ENABLED(self) -> bool: return str(self.get("AB_TESTING_ENABLED")).strip() in ("1", "true", "True")

    # Phase 5: Autonomous Behavior - Adaptive Cache Properties
    @property
    def CACHE_THRESHOLD_MIN(self) -> float: return float(self.get("CACHE_THRESHOLD_MIN", 0.85))
    @property
    def CACHE_THRESHOLD_MAX(self) -> float: return float(self.get("CACHE_THRESHOLD_MAX", 0.98))
    @property
    def CACHE_HIT_RATE_TARGET(self) -> float: return float(self.get("CACHE_HIT_RATE_TARGET", 0.20))
    @property
    def CACHE_THRESHOLD_ADAPT_ENABLED(self) -> bool: return str(self.get("CACHE_THRESHOLD_ADAPT_ENABLED")).strip() in ("1", "true", "True")

    # Phase 5: Autonomous Behavior - Predictor Validation Properties
    @property
    def PREDICTOR_VALIDATION_ENABLED(self) -> bool: return str(self.get("PREDICTOR_VALIDATION_ENABLED")).strip() in ("1", "true", "True")
    @property
    def PREDICTOR_BRIER_SCORE_THRESHOLD(self) -> float: return float(self.get("PREDICTOR_BRIER_SCORE_THRESHOLD", 0.25))
    @property
    def PREDICTOR_CALIBRATION_WINDOW(self) -> int: return int(self.get("PREDICTOR_CALIBRATION_WINDOW", 1000))

    # Phase 5: Autonomous Behavior - UQ Calibration Properties
    @property
    def UQ_CALIBRATION_ENABLED(self) -> bool: return str(self.get("UQ_CALIBRATION_ENABLED")).strip() in ("1", "true", "True")
    @property
    def UQ_QUALITY_GAP_RELAX(self) -> float: return float(self.get("UQ_QUALITY_GAP_RELAX", 0.5))
    @property
    def UQ_QUALITY_GAP_TIGHTEN(self) -> float: return float(self.get("UQ_QUALITY_GAP_TIGHTEN", 2.0))

    # Phase 5: Autonomous Behavior - Judge Calibration Properties
    @property
    def JUDGE_CALIBRATION_ENABLED(self) -> bool: return str(self.get("JUDGE_CALIBRATION_ENABLED")).strip() in ("1", "true", "True")
    @property
    def JUDGE_CACHE_AGREEMENT_TARGET(self) -> float: return float(self.get("JUDGE_CACHE_AGREEMENT_TARGET", 0.7))

    # Circuit Breaker Properties
    @property
    def CIRCUIT_BREAKER_FAIL_MAX(self) -> int: return int(self.get("CIRCUIT_BREAKER_FAIL_MAX", 5))
    @property
    def CIRCUIT_BREAKER_RESET_TIMEOUT(self) -> int: return int(self.get("CIRCUIT_BREAKER_RESET_TIMEOUT", 60))
    @property
    def CIRCUIT_BREAKER_LOCAL_FAIL_MAX(self) -> int: return int(self.get("CIRCUIT_BREAKER_LOCAL_FAIL_MAX", 3))
    @property
    def CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT(self) -> int: return int(self.get("CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT", 30))

    # Backpressure / Concurrency Control
    @property
    def MAX_CONCURRENT_REQUESTS(self) -> int: return int(self.get("MAX_CONCURRENT_REQUESTS", 500))
    @property
    def BACKPRESSURE_ENABLED(self) -> bool: return str(self.get("BACKPRESSURE_ENABLED")).strip() in ("1", "true", "True")

    # Emergency Fallback Models
    @property
    def EMERGENCY_FALLBACK_MODELS(self) -> List[str]:
        return _load_json_list(self.get("EMERGENCY_FALLBACK_MODELS", "[]"))


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
        min_timeout = int(_read("MIN_TIMEOUT", 30))
        max_timeout = int(_read("MAX_TIMEOUT", 1200))
        if min_timeout <= 0:
            errors.append("MIN_TIMEOUT must be > 0")
        if max_timeout <= 0:
            errors.append("MAX_TIMEOUT must be > 0")
        if min_timeout > max_timeout:
            errors.append("MIN_TIMEOUT must be <= MAX_TIMEOUT")
    except Exception:
        errors.append("Timeout settings are invalid")

    try:
        cq = float(_read("NSGA_W_QUALITY", 1.0))
        cl = float(_read("NSGA_W_LATENCY", 0.5))
        cc = float(_read("NSGA_W_COST", 100.0))
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
    global _reload_listener_thread
    if _reload_listener_thread and _reload_listener_thread.is_alive():
        return
    _reload_listener_stop.clear()

    # Usamos uma conexão dedicada para o listener, sem timeout
    def _bg():
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
    global _reload_listener_thread
    _reload_listener_stop.set()
    if _reload_listener_thread and _reload_listener_thread.is_alive():
        _reload_listener_thread.join(timeout=1.0)
    _reload_listener_thread = None
