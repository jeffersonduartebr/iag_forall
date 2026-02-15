# -*- coding: utf-8 -*-
"""
constants.py — Application Constants
--------------------------------------
Centralized location for all magic numbers and configuration constants.
These values can be overridden by environment variables where noted.
"""

import os

# ==============================================================================
# Rate Limiting Constants
# ==============================================================================

# Maximum requests allowed per window (default: 100)
RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))

# Time window in seconds (default: 60)
RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# Cleanup interval in seconds (default: 300 = 5 minutes)
RATE_LIMIT_CLEANUP_INTERVAL: int = int(os.getenv("RATE_LIMIT_CLEANUP_INTERVAL", "300"))

# TTL for rate limit entries in seconds (default: 3600 = 1 hour)
RATE_LIMIT_ENTRY_TTL: int = int(os.getenv("RATE_LIMIT_ENTRY_TTL", "3600"))

# ==============================================================================
# GZip Compression Constants
# ==============================================================================

# Minimum response size in bytes for GZip compression (default: 1000 = 1KB)
GZIP_MIN_SIZE: int = int(os.getenv("GZIP_MIN_SIZE", "1000"))

# ==============================================================================
# EMA (Exponential Moving Average) Cache Constants
# ==============================================================================

# Maximum entries in EMA history cache
EMA_MAX_ENTRIES: int = int(os.getenv("EMA_MAX_ENTRIES", "1000"))

# TTL for EMA cache entries in seconds (default: 86400 = 24 hours)
EMA_TTL_SECONDS: int = int(os.getenv("EMA_TTL_SECONDS", "86400"))

# EMA batch flush interval in seconds
EMA_BATCH_INTERVAL_S: int = int(os.getenv("EMA_BATCH_INTERVAL_S", "30"))

# Maximum EMA batch queue size before forced flush
EMA_BATCH_MAX_SIZE: int = int(os.getenv("EMA_BATCH_MAX_SIZE", "100"))

# EMA history log retention in days
EMA_LOG_RETENTION_DAYS: int = int(os.getenv("EMA_LOG_RETENTION_DAYS", "30"))

# ==============================================================================
# Query Log Constants
# ==============================================================================

# Query log retention in days (default: 7)
LOG_RETENTION_DAYS: int = int(os.getenv("QUERY_LOG_RETENTION_DAYS", "7"))

# ==============================================================================
# Database Pool Constants
# ==============================================================================

# Connection pool size
DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "20"))

# Maximum pool overflow connections
DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))

# Connection recycle time in seconds (default: 300 = 5 minutes)
DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "300"))

# Connection pool timeout in seconds
DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))

# ==============================================================================
# Redis Constants
# ==============================================================================

# Maximum Redis connections
REDIS_MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))

# Redis socket timeout in seconds
REDIS_SOCKET_TIMEOUT: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))

# Redis socket connect timeout in seconds
REDIS_SOCKET_CONNECT_TIMEOUT: float = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5.0"))

# Redis health check interval in seconds
REDIS_HEALTH_CHECK_INTERVAL: int = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))

# ==============================================================================
# Timeout Constants
# ==============================================================================

# Default request timeout in seconds
DEFAULT_REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))

# Base timeout for model inference
BASE_TIMEOUT: int = int(os.getenv("MIN_TIMEOUT", "60"))

# Maximum timeout for model inference
MAX_TIMEOUT: int = int(os.getenv("MAX_TIMEOUT", "1200"))

# Timeout multiplier for large models
TIMEOUT_MULTIPLIER: float = float(os.getenv("ADAPTIVE_TIMEOUT_MULTIPLIER", "2.0"))

# Timeout multiplier for reasoning models
REASONING_TIMEOUT_MULTIPLIER: float = float(os.getenv("ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER", "3.0"))

# ==============================================================================
# Token Counting Constants
# ==============================================================================

# Maximum size for token encoder cache
TOKEN_ENCODER_CACHE_SIZE: int = int(os.getenv("TOKEN_ENCODER_CACHE_SIZE", "128"))

# ==============================================================================
# Bandit Algorithm Constants
# ==============================================================================

# Default epsilon for epsilon-greedy bandit
DEFAULT_BANDIT_EPSILON: float = float(os.getenv("BANDIT_EPSILON", "0.12"))

# Number of centroids for semantic clustering
CENTROIDS_K: int = int(os.getenv("CENTROIDS_K", "20"))

# Embedding dimension for centroids
CENTROIDS_DIM: int = int(os.getenv("CENTROIDS_DIM", "768"))

# Learning rate for centroid updates
CENTROIDS_LEARN_RATE: float = float(os.getenv("CENTROIDS_LEARN_RATE", "0.15"))

# Minimum similarity for creating new centroid
CENTROIDS_MIN_SIM_CREATE: float = float(os.getenv("CENTROIDS_MIN_SIM_CREATE", "0.35"))

# ==============================================================================
# Cache Constants
# ==============================================================================

# Semantic cache similarity threshold
CACHE_THRESHOLD: float = float(os.getenv("CACHE_THRESHOLD", "0.92"))

# Cache TTL in days
CACHE_TTL_DAYS: int = int(os.getenv("CACHE_TTL_DAYS", "7"))

# ==============================================================================
# Pricing Cache Constants
# ==============================================================================

# Local pricing cache TTL in seconds
PRICING_CACHE_TTL: int = int(os.getenv("PRICING_CACHE_TTL", "300"))

# Redis pricing cache TTL in seconds
PRICING_REDIS_TTL: int = int(os.getenv("PRICING_REDIS_TTL", "3600"))

# ==============================================================================
# Ollama Constants
# ==============================================================================

# Ollama concurrency limit
OLLAMA_CONCURRENCY_LIMIT: int = int(os.getenv("OLLAMA_CONCURRENCY_LIMIT", "30"))

# Ollama model pull timeout in seconds
OLLAMA_PULL_TIMEOUT: int = int(os.getenv("OLLAMA_PULL_TIMEOUT", "900"))

# ==============================================================================
# Judge Constants
# ==============================================================================

# Minimum judge sampling rate
JUDGE_MIN_SAMPLE_RATE: float = float(os.getenv("JUDGE_MIN_SAMPLE_RATE", "0.05"))

# Judge timeout in seconds
JUDGE_TIMEOUT_S: int = int(os.getenv("JUDGES_TIMEOUT_S", "15"))

# ==============================================================================
# Metrics Constants
# ==============================================================================

# DB pool metrics update interval in seconds
DB_POOL_METRICS_INTERVAL: int = int(os.getenv("DB_POOL_METRICS_INTERVAL", "30"))
