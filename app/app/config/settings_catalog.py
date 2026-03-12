# Objective: Configuration support code for settings catalog.
"""Catalog of dynamic settings grouped by domain."""

from __future__ import annotations

import json
from typing import Dict, List


SETTINGS_BY_DOMAIN: Dict[str, Dict[str, str]] = {
    "runtime": {
        "MAX_TOKENS_DEFAULT": "2000",
        "TEMPERATURE_DEFAULT": "0.55",
        "QUERY_LOG_RETENTION_DAYS": "7",
        "REQUEST_TIMEOUT_SECONDS": "120",
        "REQUEST_DEDUP_ENABLED": "1",
        "MAX_CONCURRENT_REQUESTS": "500",
        "BACKPRESSURE_ENABLED": "1",
    },
    "redis": {
        "REDIS_HOST": "redis",
        "REDIS_PORT": "6379",
        "REDIS_DB": "0",
        "REDIS_PASSWORD": "",
    },
    "embeddings": {
        "EMBED_MODEL": "nomic-embed-text",
        "EMBED_PROVIDER": "ollama",
        "EMBED_DEVICE": "cpu",
        "EMBED_TEXT_MODEL": "nomic-embed-text",
        "TEXT_EMBEDDING_MODEL": "nomic-embed-text",
        "IMAGE_EMBEDDING_MODEL": "clip-vit-large-patch14",
        "MULTIMODAL_EMBEDDING_MODEL": "clip-vit-large-patch14",
    },
    "centroids": {
        "CENTROIDS_DIM": "768",
        "CENTROIDS_K": "20",
        "CENTROIDS_MIN_SIM_CREATE": "0.35",
        "CENTROIDS_ENABLE_ONLINE": "1",
        "CENTROIDS_UPDATE_INTERVAL_S": "1800",
        "CENTROIDS_MIN_RECORDS_FOR_TRAIN": "50",
        "CENTROIDS_MAX_HISTORY": "50000",
        "CENTROIDS_HOURLY_REFRESH_ENABLED": "1",
        "CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH": "50",
    },
    "judges": {
        "JUDGES_ENABLED": "1",
        "JUDGES_MODE": "llm",
        "JUDGES_LOCAL_MODEL": "ollama/phi4:latest",
        "JUDGES_REMOTE_MODEL": "ollama/phi4:latest",
        "JUDGES_TIMEOUT_S": "15",
        "JUDGE_MIN_SAMPLE_RATE": "0.05",
        "JUDGE_CALIBRATION_ENABLED": "1",
        "JUDGE_CACHE_AGREEMENT_TARGET": "0.7",
        "JUDGE_MODELS": "[]",
    },
    "providers": {
        "OLLAMA_BASE_URL": "http://ollama:11434",
        "OLLAMA_HOST": "http://ollama:11434",
        "OLLAMA_CONCURRENCY_LIMIT": "5",
        "CANDIDATE_MODELS_LIST": "[]",
        "CANDIDATE_VISION_MODELS_LIST": "[]",
        "CANDIDATE_MULTIMODAL_MODELS_LIST": "[]",
        "VLM_OLLAMA_MODELS": json.dumps(
            [
                "qwen3-vl:8b",
                "gemma3:4b",
                "llama3.2:3b",
                "llama3:8b",
                "llava:7b",
                "llama3.2-vision:11b",
                "llava-llama3:8b",
                "granite3.2-vision:2b",
            ]
        ),
        "EMERGENCY_FALLBACK_MODELS": json.dumps(
            [
                "ollama/phi4:latest",
                "ollama/gemma3:4b",
                "ollama/llama3:8b",
            ]
        ),
    },
    "rag_cache": {
        "CACHE_TTL_DAYS": "7",
        "CACHE_THRESHOLD": "0.92",
        "CACHE_THRESHOLD_MIN": "0.85",
        "CACHE_THRESHOLD_MAX": "0.98",
        "CACHE_HIT_RATE_TARGET": "0.20",
        "CACHE_THRESHOLD_ADAPT_ENABLED": "0",
        "UNCERTAINTY_THRESHOLD": "0.7",
        "RERANK_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "RERANK_ENABLED": "1",
        "RAG_DATA_DIR": "/app/data",
    },
    "routing": {
        "BANDIT_EPSILON": "0.12",
        "NSGA_W_QUALITY": "1.0",
        "NSGA_W_LATENCY": "0.5",
        "NSGA_W_COST": "100.0",
        "NSGA_W_ALIGNMENT": "1.0",
        "NSGA_CONVERGENCE_HISTORY_SIZE": "20",
        "NSGA_UPDATE_INTERVAL_S": "300",
        "NSGA_LOOKBACK_MINUTES": "180",
        "NSGA_LOOKBACK_MAXROWS": "2000",
    },
    "resilience": {
        "CASCADE_WARNING_THRESHOLD": "0.3",
        "CASCADE_CRITICAL_THRESHOLD": "0.5",
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
        "CIRCUIT_BREAKER_FAIL_MAX": "5",
        "CIRCUIT_BREAKER_RESET_TIMEOUT": "60",
        "CIRCUIT_BREAKER_LOCAL_FAIL_MAX": "3",
        "CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT": "30",
    },
    "feedback": {
        "DRIFT_THRESHOLD": "0.15",
        "DRIFT_WINDOW_SIZE": "100",
        "USER_FEEDBACK_WEIGHT": "0.7",
    },
    "experimentation": {
        "AB_TESTING_ENABLED": "0",
        "META_OPT_ENABLED": "0",
        "META_OPT_SCHEDULE_HOUR": "3",
        "META_OPT_SCHEDULED_TRIALS": "20",
        "METAOPT_REPS": "5",
        "METAOPT_TRIALS": "100",
    },
    "prediction": {
        "PREDICTOR_VALIDATION_ENABLED": "1",
        "PREDICTOR_BRIER_SCORE_THRESHOLD": "0.25",
        "PREDICTOR_CALIBRATION_WINDOW": "1000",
        "UQ_CALIBRATION_ENABLED": "1",
        "UQ_QUALITY_GAP_RELAX": "0.5",
        "UQ_QUALITY_GAP_TIGHTEN": "2.0",
    },
}

SETTINGS_DEFAULTS: Dict[str, str] = {
    key: value
    for domain_defaults in SETTINGS_BY_DOMAIN.values()
    for key, value in domain_defaults.items()
}

REQUIRES_RESTART_KEYS = {
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_DB",
    "REDIS_PASSWORD",
    "OLLAMA_HOST",
    "OLLAMA_BASE_URL",
    "EMBED_MODEL",
    "EMBED_PROVIDER",
    "EMBED_DEVICE",
    "TEXT_EMBEDDING_MODEL",
    "IMAGE_EMBEDDING_MODEL",
    "MULTIMODAL_EMBEDDING_MODEL",
    "RAG_DATA_DIR",
}

SETTING_METADATA: Dict[str, Dict[str, str]] = {
    key: {
        "domain": domain,
        "mutability": "requires_restart" if key in REQUIRES_RESTART_KEYS else "runtime_safe",
    }
    for domain, domain_defaults in SETTINGS_BY_DOMAIN.items()
    for key in domain_defaults
}


def known_setting_keys() -> List[str]:
    """Return catalogued setting keys in stable order."""
    return list(SETTINGS_DEFAULTS.keys())


def metadata_for(key: str) -> Dict[str, str]:
    """Return metadata for one setting key."""
    return dict(SETTING_METADATA.get(key, {}))


def is_known_setting(key: str) -> bool:
    """Return whether one key belongs to the dynamic settings catalog."""
    return key in SETTING_METADATA


def is_runtime_mutable(key: str) -> bool:
    """Return whether one setting can be applied without restart."""
    return metadata_for(key).get("mutability") == "runtime_safe"
