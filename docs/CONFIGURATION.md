# Configuration Guide

This document describes all configuration options for the Multi-Objective LLM Router.

## Table of Contents

1. [Configuration Layers](#configuration-layers)
2. [Core Settings](#core-settings)
3. [Provider Settings](#provider-settings)
4. [Router & Optimization](#router--optimization)
5. [Resilience Settings](#resilience-settings)
6. [Cache & RAG](#cache--rag)
7. [Monitoring & Observability](#monitoring--observability)

---

## Configuration Layers

Settings are loaded in priority order:

1. **Environment Variables** (highest priority)
2. **Redis Cache** (30-second TTL)
3. **MariaDB Table** (`settings_dynamic`)
4. **Code Defaults** (lowest priority)

### Dynamic Updates

Settings can be updated at runtime via:

```bash
# Admin API
curl -X PUT http://localhost:8000/admin/settings \
  -H "X-Admin-Token: your-token" \
  -H "Content-Type: application/json" \
  -d '{"BANDIT_EPSILON": "0.15"}'
```

Changes propagate via Redis pub/sub for hot-reload.

---

## Core Settings

### Database Connection

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `mariadb` | MariaDB host |
| `DB_PORT` | `3306` | MariaDB port |
| `DB_USER` | `router_user` | Database username |
| `DB_PASS` | (required) | Database password |
| `DB_NAME` | `routerdb` | Database name |

### Redis Connection

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `redis` | Redis host |
| `REDIS_PORT` | `6378` | Redis port |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_PASSWORD` | (empty) | Redis password |

### Admin & Security

| Variable | Default | Description |
|----------|---------|-------------|
| `ADMIN_TOKEN` | `changeme-please` | Token for admin endpoints |

---

## Provider Settings

### Ollama (Local)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama server URL |
| `OLLAMA_CONCURRENCY_LIMIT` | `30` | Max concurrent requests |

### OpenAI

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |

### Anthropic

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key |

### Google Gemini

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Google Gemini API key |

### Model Lists

| Variable | Default | Description |
|----------|---------|-------------|
| `CANDIDATE_MODELS_LIST` | `[]` | Text model candidates (JSON array) |
| `CANDIDATE_VISION_MODELS_LIST` | `[]` | Vision model candidates |
| `CANDIDATE_MULTIMODAL_MODELS_LIST` | `[]` | Multimodal model candidates |
| `VLM_OLLAMA_MODELS` | (see below) | Ollama VLM models to preload |
| `JUDGE_MODELS` | `[]` | Models available for judging |

Default VLM models:
```json
["qwen3-vl:8b", "gemma3:4b", "llama3.2:3b", "llama3:8b", "llava:7b", "llama3.2-vision:11b"]
```

---

## Router & Optimization

### Request Handling

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_TOKENS_DEFAULT` | `2000` | Default max tokens for responses |
| `TEMPERATURE_DEFAULT` | `0.55` | Default temperature |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Global request timeout |
| `REQUEST_DEDUP_ENABLED` | `1` | Enable request deduplication |

### NSGA-II Weights

| Variable | Default | Description |
|----------|---------|-------------|
| `NSGA_W_QUALITY` | `1.0` | Quality objective weight |
| `NSGA_W_LATENCY` | `0.5` | Latency objective weight |
| `NSGA_W_COST` | `100.0` | Cost objective weight |
| `NSGA_W_ALIGNMENT` | `1.0` | Alignment objective weight |

#### Configuration Profiles

**Quality-Focused:**
```env
NSGA_W_QUALITY=2.0
NSGA_W_LATENCY=0.3
NSGA_W_COST=50.0
```

**Cost-Focused:**
```env
NSGA_W_QUALITY=0.8
NSGA_W_LATENCY=0.5
NSGA_W_COST=200.0
```

**Latency-Focused:**
```env
NSGA_W_QUALITY=0.8
NSGA_W_LATENCY=1.5
NSGA_W_COST=50.0
```

### NSGA-II Optimizer

| Variable | Default | Description |
|----------|---------|-------------|
| `NSGA_UPDATE_INTERVAL_S` | `300` | Weight update interval (seconds) |
| `NSGA_LOOKBACK_MINUTES` | `180` | Query history lookback |
| `NSGA_LOOKBACK_MAXROWS` | `2000` | Max rows to analyze |
| `NSGA_CONVERGENCE_HISTORY_SIZE` | `20` | Convergence check window |

### Multi-Armed Bandits

| Variable | Default | Description |
|----------|---------|-------------|
| `BANDIT_EPSILON` | `0.12` | Exploration rate (ε-greedy) |

### Uncertainty Quantification

| Variable | Default | Description |
|----------|---------|-------------|
| `UNCERTAINTY_THRESHOLD` | `0.7` | High uncertainty threshold |
| `UQ_CALIBRATION_ENABLED` | `1` | Auto-adjust UQ thresholds |
| `UQ_QUALITY_GAP_RELAX` | `0.5` | Gap for relaxing threshold |
| `UQ_QUALITY_GAP_TIGHTEN` | `2.0` | Gap for tightening threshold |

### Risk Factors

| Variable | Default | Description |
|----------|---------|-------------|
| `RISK_FACTOR_SOTA_HIGH_UQ` | `1.3` | SOTA boost for high UQ |
| `RISK_FACTOR_LOCAL_HIGH_UQ` | `0.6` | Local penalty for high UQ |
| `RISK_FACTOR_LOCAL_LOW_UQ` | `1.1` | Local boost for low UQ |
| `RISK_FACTOR_ADAPT_ENABLED` | `0` | Enable adaptive risk factors |
| `RISK_FACTOR_ADAPT_RATE` | `0.02` | Adaptation rate |

### Adaptive Timeout

| Variable | Default | Description |
|----------|---------|-------------|
| `ADAPTIVE_TIMEOUT_ENABLED` | `0` | Enable adaptive timeouts |
| `ADAPTIVE_TIMEOUT_MULTIPLIER` | `2.0` | Standard model multiplier |
| `ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER` | `3.0` | Reasoning model multiplier |
| `MIN_TIMEOUT` | `30` | Minimum timeout (seconds) |
| `MAX_TIMEOUT` | `1200` | Maximum timeout (seconds) |

### Quality Judges

| Variable | Default | Description |
|----------|---------|-------------|
| `JUDGES_ENABLED` | `1` | Enable quality judging |
| `JUDGES_MODE` | `llm` | Judge mode (llm, heuristic) |
| `JUDGES_LOCAL_MODEL` | `ollama/phi4:latest` | Local judge model |
| `JUDGES_REMOTE_MODEL` | `gpt-5-mini` | Remote judge model |
| `JUDGES_TIMEOUT_S` | `15` | Judge timeout |
| `JUDGE_MIN_SAMPLE_RATE` | `0.05` | Minimum sampling rate (5%) |
| `JUDGE_CALIBRATION_ENABLED` | `1` | Enable judge calibration |
| `JUDGE_CACHE_AGREEMENT_TARGET` | `0.7` | Target cache agreement |

### Online Learning

| Variable | Default | Description |
|----------|---------|-------------|
| `PREDICTOR_VALIDATION_ENABLED` | `1` | Enable predictor validation |
| `PREDICTOR_BRIER_SCORE_THRESHOLD` | `0.25` | Max acceptable Brier score |
| `PREDICTOR_CALIBRATION_WINDOW` | `1000` | Calibration sample window |

### Meta Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `META_OPT_ENABLED` | `0` | Enable Bayesian meta-optimization |
| `META_OPT_SCHEDULE_HOUR` | `3` | Scheduled optimization hour |
| `META_OPT_SCHEDULED_TRIALS` | `20` | Trials per scheduled run |
| `METAOPT_REPS` | `5` | Repetitions per trial |
| `METAOPT_TRIALS` | `100` | Max Optuna trials |

---

## Resilience Settings

### Circuit Breakers

| Variable | Default | Description |
|----------|---------|-------------|
| `CIRCUIT_BREAKER_FAIL_MAX` | `5` | Failures before open (cloud) |
| `CIRCUIT_BREAKER_RESET_TIMEOUT` | `60` | Reset timeout seconds (cloud) |
| `CIRCUIT_BREAKER_LOCAL_FAIL_MAX` | `3` | Failures before open (local) |
| `CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT` | `30` | Reset timeout seconds (local) |

### Backpressure

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_REQUESTS` | `500` | Max concurrent requests |
| `BACKPRESSURE_ENABLED` | `1` | Enable backpressure control |

### Emergency Fallback

| Variable | Default | Description |
|----------|---------|-------------|
| `EMERGENCY_FALLBACK_MODELS` | (see below) | Fallback models for cascade failure |
| `CASCADE_WARNING_THRESHOLD` | `0.3` | Warning at 30% models failed |
| `CASCADE_CRITICAL_THRESHOLD` | `0.5` | Critical at 50% models failed |

Default emergency fallbacks:
```json
["ollama/phi4:latest", "ollama/gemma3:4b", "ollama/llama3:8b"]
```

---

## Cache & RAG

### Semantic Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_TTL_DAYS` | `7` | Cache entry TTL |
| `CACHE_THRESHOLD` | `0.92` | Similarity threshold for cache hit |
| `CACHE_THRESHOLD_MIN` | `0.85` | Minimum threshold (adaptive) |
| `CACHE_THRESHOLD_MAX` | `0.98` | Maximum threshold (adaptive) |
| `CACHE_HIT_RATE_TARGET` | `0.20` | Target cache hit rate |
| `CACHE_THRESHOLD_ADAPT_ENABLED` | `0` | Enable adaptive threshold |

### RAG Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_DATA_DIR` | `/app/data` | RAG documents directory |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `RERANK_ENABLED` | `1` | Enable reranking |

### Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `nomic-embed-text` | Default embedding model |
| `EMBED_PROVIDER` | `ollama` | Embedding provider |
| `EMBED_DEVICE` | `cpu` | Embedding device |
| `EMBED_TEXT_MODEL` | `nomic-embed-text` | Text embedding model |
| `TEXT_EMBEDDING_MODEL` | `nomic-embed-text` | Alias for text embeddings |
| `IMAGE_EMBEDDING_MODEL` | `clip-vit-large-patch14` | Image embedding model |
| `MULTIMODAL_EMBEDDING_MODEL` | `gpt-4o-mini-embed` | Multimodal embeddings |

### Centroids (Clustering)

| Variable | Default | Description |
|----------|---------|-------------|
| `CENTROIDS_DIM` | `768` | Embedding dimension |
| `CENTROIDS_K` | `20` | Number of clusters |
| `CENTROIDS_MIN_SIM_CREATE` | `0.35` | Min similarity to create |
| `CENTROIDS_ENABLE_ONLINE` | `1` | Enable online updates |
| `CENTROIDS_UPDATE_INTERVAL_S` | `1800` | Update interval |
| `CENTROIDS_MIN_RECORDS_FOR_TRAIN` | `50` | Min records to train |
| `CENTROIDS_MAX_HISTORY` | `50000` | Max history records |
| `CENTROIDS_HOURLY_REFRESH_ENABLED` | `1` | Enable hourly refresh |
| `CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH` | `50` | Min rows for refresh |

---

## Monitoring & Observability

### Query Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `QUERY_LOG_RETENTION_DAYS` | `7` | Days to retain query logs |

### Settings Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `SETTINGS_CACHE_SIZE` | `512` | LRU cache max entries |
| `SETTINGS_CACHE_TTL_S` | `30` | Cache TTL seconds |

### User Feedback

| Variable | Default | Description |
|----------|---------|-------------|
| `DRIFT_THRESHOLD` | `0.15` | Quality drift detection |
| `DRIFT_WINDOW_SIZE` | `100` | Drift detection window |
| `USER_FEEDBACK_WEIGHT` | `0.7` | User vs judge weight |

### A/B Testing

| Variable | Default | Description |
|----------|---------|-------------|
| `AB_TESTING_ENABLED` | `0` | Enable A/B testing |

---

## Environment Files

### Development (.env.dev)

```env
# Database
DB_HOST=localhost
DB_PORT=3307
DB_USER=router_user
DB_PASS=dev_password
DB_NAME=routerdb

# Redis
REDIS_HOST=localhost
REDIS_PORT=6378
REDIS_PASSWORD=dev_redis

# Ollama
OLLAMA_HOST=http://localhost:11434

# Debug
ADMIN_TOKEN=dev-token
ENABLE_SMOKE_TESTS=1

# Lower thresholds for testing
BANDIT_EPSILON=0.2
JUDGE_MIN_SAMPLE_RATE=0.1
```

### Production (.env.prod)

```env
# Database
DB_HOST=mariadb
DB_PORT=3306
DB_USER=router_user
DB_PASS=${DB_PASSWORD}
DB_NAME=routerdb

# Redis
REDIS_HOST=redis
REDIS_PORT=6378
REDIS_PASSWORD=${REDIS_PASSWORD}

# Providers
OLLAMA_HOST=http://ollama:11434
OPENAI_API_KEY=${OPENAI_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_KEY}
GEMINI_API_KEY=${GEMINI_KEY}

# Security
ADMIN_TOKEN=${ADMIN_SECRET}

# Production tuning
MAX_CONCURRENT_REQUESTS=1000
BACKPRESSURE_ENABLED=1
REQUEST_TIMEOUT_SECONDS=180
ADAPTIVE_TIMEOUT_ENABLED=1

# Cost optimization
NSGA_W_COST=150.0
JUDGE_MIN_SAMPLE_RATE=0.03
```

---

## Validation

### Check Configuration

```bash
# View current settings
curl http://localhost:8000/admin/settings \
  -H "X-Admin-Token: your-token"

# Health check (includes component status)
curl http://localhost:8000/health
```

### Common Issues

1. **Missing API Keys**: Check provider-specific environment variables
2. **Connection Refused**: Verify host/port settings match Docker network
3. **Timeout Errors**: Increase `REQUEST_TIMEOUT_SECONDS` or enable adaptive timeout
4. **Memory Issues**: Reduce `CENTROIDS_MAX_HISTORY` or `SETTINGS_CACHE_SIZE`
