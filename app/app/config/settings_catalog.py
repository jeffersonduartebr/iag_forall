# Objective: Configuration support code for settings catalog.
"""Catalog of dynamic settings grouped by domain."""

from __future__ import annotations

import json
from typing import Dict, List

SETTINGS_BY_DOMAIN: Dict[str, Dict[str, str]] = {
    "auth": {
        "REQUIRE_API_AUTH": "0",
        "API_KEYS": "",
        "JWT_SECRET": "",
        "JWT_ALGORITHM": "HS256",
        "ENV": "development",
        "METRICS_TOKEN": "",
        "TRUSTED_PROXY_IPS": "",
        "TRUST_HEADER_ROLES": "0",
        "ENFORCE_TENANT_BINDING": "1",
        "ROADMAP_AUTO_DDL": "1",
    },
    "runtime": {
        "MAX_TOKENS_DEFAULT": "2000",
        "ROUTER_SIMPLE_QUERY_MAX_TOKENS": "512",
        "ROUTER_SIMPLE_TEXT_MAX_FALLBACKS": "1",
        "TEMPERATURE_DEFAULT": "0.55",
        "QUERY_LOG_RETENTION_DAYS": "7",
        "REQUEST_TIMEOUT_SECONDS": "120",
        "REQUEST_DEDUP_ENABLED": "1",
        "MAX_CONCURRENT_REQUESTS": "500",
        "BACKPRESSURE_ENABLED": "1",
        "BACKPRESSURE_REDIS_ENABLED": "0",
        "REDIS_REQUIRED_IN_PRODUCTION": "1",
        "REDIS_DEDUP_ENABLED": "1",
        "ADAPTIVE_LIMITER_ENABLED": "1",
        "ADAPTIVE_LIMITER_WINDOW_SECONDS": "15",
        "ADAPTIVE_LIMITER_HYSTERESIS_WINDOWS": "3",
        "ADAPTIVE_LIMITER_ELEVATED_UTILIZATION": "0.80",
        "ADAPTIVE_LIMITER_CONGESTED_UTILIZATION": "1.00",
        "ADAPTIVE_LIMITER_ELEVATED_QUEUE_WAIT_P95_MS": "500",
        "ADAPTIVE_LIMITER_CONGESTED_QUEUE_WAIT_P95_MS": "1000",
        "ADAPTIVE_LIMITER_SYNC_QUEUE_WAIT_MS": "250",
        "ADAPTIVE_LIMITER_INTERACTIVE_PER_SLOT_ELEVATED": "12",
        "ADAPTIVE_LIMITER_INTERACTIVE_PER_SLOT_CONGESTED": "6",
        "ADAPTIVE_LIMITER_ADMIN_PER_SLOT_ELEVATED": "3",
        "ADAPTIVE_LIMITER_ADMIN_PER_SLOT_CONGESTED": "1",
        "QUERY_JOB_TTL_SECONDS": "3600",
        "QUERY_JOB_MAX_PENDING_PER_TENANT": "500",
        "QUERY_JOB_MAX_PENDING_PER_IP": "250",
        "SYNC_DEADLINE_SIMPLE_SECONDS": "25",
        "SYNC_DEADLINE_KNOWLEDGE_SECONDS": "40",
        "SYNC_DEADLINE_REASONING_SECONDS": "70",
        "SYNC_DEADLINE_VISION_SECONDS": "100",
        "PROVIDER_TIMEOUT_SIMPLE_SECONDS": "20",
        "PROVIDER_TIMEOUT_KNOWLEDGE_SECONDS": "35",
        "PROVIDER_TIMEOUT_REASONING_SECONDS": "60",
        "PROVIDER_TIMEOUT_VISION_SECONDS": "90",
        "ROUTER_PERF_MODE": "0",
        "ROUTER_QUERY_CLASSIFIER_ENABLED": "1",
    },
    "redis": {
        "REDIS_HOST": "redis",
        "REDIS_PORT": "6379",
        "REDIS_DB": "0",
        "REDIS_PASSWORD": "",
        "TENANT_RATE_LIMIT_ENABLED": "1",
        "TENANT_RATE_LIMIT_RPM": "120",
        "TENANT_RATE_LIMIT_WINDOW_S": "60",
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
        "OPENROUTER_API_KEY": "",
        "OLLAMA_BASE_URL": "http://ollama:11434",
        "OLLAMA_HOST": "http://ollama:11434",
        "OLLAMA_CONCURRENCY_LIMIT": "5",
        "OLLAMA_DYNAMIC_CONCURRENCY_ENABLED": "0",
        "OLLAMA_CONCURRENCY_MIN": "1",
        "OLLAMA_CONCURRENCY_MAX": "5",
        "OLLAMA_VRAM_TARGET_UTILIZATION": "0.72",
        "OLLAMA_VRAM_HIGH_WATERMARK": "0.82",
        "OLLAMA_VRAM_LOW_WATERMARK": "0.55",
        "OLLAMA_VRAM_POLL_INTERVAL_SECONDS": "5",
        "OLLAMA_CONCURRENCY_STEP_UP": "1",
        "OLLAMA_CONCURRENCY_STEP_DOWN": "1",
        "OLLAMA_CONCURRENCY_STABLE_WINDOWS": "3",
        "OLLAMA_GPU_INDEX": "0",
        "OLLAMA_WARM_MODELS": "[]",
        "OLLAMA_INTERACTIVE_WARM_MODELS": "[]",
        "OLLAMA_INTERACTIVE_MODEL_SET": "[]",
        "OLLAMA_WARMUP_GENERATE_ENABLED": "1",
        "OLLAMA_LOAD_PENALTY_ENABLED": "1",
        "OLLAMA_BACKGROUND_LOAD_SHEDDING_ENABLED": "1",
        "OLLAMA_ROUTE_CANDIDATE_LIMIT": "3",
        "PROVIDER_UNAVAILABLE_NEGATIVE_CACHE_TTL_SECONDS": "30",
        "MODEL_UNAVAILABLE_NEGATIVE_CACHE_TTL_SECONDS": "30",
        "CANDIDATE_MODELS_LIST": "[]",
        "CANDIDATE_VISION_MODELS_LIST": "[]",
        "CANDIDATE_MULTIMODAL_MODELS_LIST": "[]",
        "CANDIDATE_TOOL_MODELS_LIST": "[]",
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
        # Canonicalize queries (casefold + whitespace collapse) before hashing/
        # embedding so trivial surface variants share a cache entry (perf #24).
        "SEMANTIC_CACHE_NORMALIZE_ENABLED": "1",
        "UNCERTAINTY_THRESHOLD": "0.7",
        "RAG_SIMPLE_QUERY_BYPASS_ENABLED": "1",
        "RAG_LIGHT_TOP_K": "2",
        "RAG_LIGHT_VECTOR_TOP_K": "6",
        "RAG_LIGHT_SPARSE_TOP_K": "6",
        "RAG_RERANK_MIN_CANDIDATES": "3",
        "RAG_LIGHT_CONTEXT_TOKEN_BUDGET": "480",
        "RAG_FULL_CONTEXT_TOKEN_BUDGET": "1200",
        "RAG_CONTEXT_TOKEN_BUDGET": "1200",
        "RAG_CONTEXT_QUALITY_MIN_DOCS": "2",
        "RERANK_ENABLED_FOR_LIGHT_RETRIEVAL": "0",
        "RERANK_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "RERANK_ENABLED": "1",
        "RAG_DATA_DIR": "/app/data",
        "CHROMA_HOST": "",
        "CHROMA_PORT": "8000",
        "CHROMA_PATH": "/data/chroma",
    },
    "routing": {
        "BANDIT_EPSILON": "0.12",
        "OPENROUTER_EXPLORATION_ENABLED": "0",
        "OPENROUTER_EXPLORATION_MODE": "balanced",
        "OPENROUTER_EXPLORATION_RATE": "0.10",
        "OPENROUTER_EXPLORATION_MAX_PER_DAY": "100",
        "OPENROUTER_EXPLORATION_MAX_USD_PER_DAY": "3.0",
        "OPENROUTER_EXPLORATION_MAX_PRICE_PROMPT_1K": "0.01",
        "OPENROUTER_EXPLORATION_MAX_PRICE_COMPLETION_1K": "0.03",
        "OPENROUTER_EXPLORATION_PROMOTE_MIN_SAMPLES": "15",
        "OPENROUTER_EXPLORATION_PROMOTE_MIN_REWARD": "0.72",
        "OPENROUTER_EXPLORATION_PROMOTE_MAX_LATENCY_S": "30.0",
        "OPENROUTER_EXPLORATION_PROMOTE_MAX_COST_USD_PER_1K": "0.02",
        "OPENROUTER_EXPLORATION_PROMOTE_MAX_FAILURE_RATE": "0.20",
        "OPENROUTER_EXPLORATION_AUTO_PROMOTE_ENABLED": "1",
        "OPENROUTER_EXPLORATION_ADAPTIVE_RATE_ENABLED": "1",
        "OPENROUTER_EXPLORATION_POOL_CACHE_TTL_S": "600",
        "OPENROUTER_EXPLORATION_SHADOW_COMPARE_RATE": "0.05",
        "OPENROUTER_EXPLORATION_CONSECUTIVE_FAILURE_BLOCK": "3",
        "OPENROUTER_EXPLORATION_PROVIDER_ALLOWLIST": json.dumps(
            ["anthropic", "openai", "google", "meta-llama", "mistralai"]
        ),
        "OPENROUTER_EXPLORATION_POOL_SIZE": "40",
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
        # Speculative (hedged) requests (perf #22): race the primary against one
        # backup to cut tail latency. Off by default (extra provider cost on the
        # hedged fraction). Delay is EMA*factor unless a fixed ms is set.
        "REQUEST_HEDGING_ENABLED": "0",
        "REQUEST_HEDGE_DELAY_MS": "0",
        "REQUEST_HEDGE_EMA_FACTOR": "1.3",
        "REQUEST_HEDGE_MAX_PARALLEL": "2",
        "CIRCUIT_BREAKER_FAIL_MAX": "5",
        "CIRCUIT_BREAKER_RESET_TIMEOUT": "60",
        "CIRCUIT_BREAKER_LOCAL_FAIL_MAX": "3",
        "CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT": "30",
    },
    "adversarial_governance": {
        # Closed-loop adversarial governance (roadmap #17). Off by default so
        # production routing is unaffected until explicitly enabled.
        "ADVGOV_ENABLED": "0",
        # Judge score (0..10) below which an adversarial attack counts as a success.
        "ADVGOV_FAIL_SCORE_THRESHOLD": "7.0",
        # Minimum duels observed before a knowledge cluster can be flagged high-risk.
        "ADVGOV_CLUSTER_MIN_SAMPLES": "5",
        # Attack-success rate at/above which a cluster is considered high-risk.
        "ADVGOV_CLUSTER_FAILURE_RATE_THRESHOLD": "0.5",
        # TTL (seconds) for per-cluster risk state in Redis (default 30 days).
        "ADVGOV_CLUSTER_TTL_S": "2592000",
        # Whether high-risk clusters / high UQ escalate the chosen model.
        "ADVGOV_ESCALATION_ENABLED": "1",
        # Preferred escalation targets (JSON list); first match in candidates wins.
        "ADVGOV_ESCALATION_MODELS": json.dumps([]),
    },
    "feedback": {
        "DRIFT_THRESHOLD": "0.15",
        "DRIFT_WINDOW_SIZE": "100",
        "USER_FEEDBACK_WEIGHT": "0.7",
        "JUDGE_FEEDBACK_MIN_SAMPLES": "30",
        "JUDGE_FEEDBACK_ERROR_THRESHOLD": "5.0",
        "JUDGE_BACKGROUND_THROTTLE_ENABLED": "1",
        "BACKGROUND_THROTTLE_INFLIGHT_THRESHOLD": "4",
        "BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS": "750",
        "QUERY_JOB_BACKGROUND_THROTTLE_PENDING_THRESHOLD": "1",
    },
    "experimentation": {
        "AB_TESTING_ENABLED": "0",
        "META_OPT_ENABLED": "0",
        "META_OPT_SCHEDULE_HOUR": "3",
        "META_OPT_SCHEDULED_TRIALS": "20",
        "METAOPT_SCHEDULED_REPS": "2",
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
    key: value for domain_defaults in SETTINGS_BY_DOMAIN.values() for key, value in domain_defaults.items()
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
    "CHROMA_HOST",
    "CHROMA_PORT",
    "CHROMA_PATH",
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
