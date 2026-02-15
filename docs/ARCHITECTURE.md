# Architecture Documentation

This document describes the architecture of the Multi-Objective LLM Router system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Diagram](#component-diagram)
3. [Query Flow](#query-flow)
4. [Feedback Loop](#feedback-loop)
5. [Data Flow](#data-flow)
6. [Service Architecture](#service-architecture)

---

## System Overview

The Multi-Objective LLM Router is a sophisticated orchestration system that intelligently routes queries to the most appropriate LLM based on three competing objectives: **cost**, **latency**, and **quality**.

### Key Capabilities

- **Multi-Provider Support**: Ollama, OpenAI, Anthropic, Google Gemini
- **Multi-Objective Optimization**: NSGA-II for Pareto-optimal routing
- **Adaptive Learning**: Multi-armed bandits with online ML
- **Multimodal Support**: Text, vision, and multimodal queries
- **RAG Integration**: Local vector store with reranking
- **Enterprise Features**: Rate limiting, circuit breakers, backpressure

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Web App   │  │    CLI      │  │   Mobile    │  │   SDK       │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
└─────────┼────────────────┼────────────────┼────────────────┼────────────────┘
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FastAPI Application (main.py)                                       │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │   │
│  │  │ GZip     │ │ Rate     │ │ Back-    │ │ Correla- │               │   │
│  │  │ Compress │ │ Limit    │ │ pressure │ │ tion ID  │               │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
┌──────────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│    ROUTER CORE       │ │   RAG ENGINE     │ │   ADMIN ENDPOINTS    │
│  ┌────────────────┐  │ │ ┌──────────────┐ │ │ ┌──────────────────┐ │
│  │ Cache Check    │  │ │ │ ChromaDB     │ │ │ │ Settings CRUD    │ │
│  │ UQ Score       │  │ │ │ Embeddings   │ │ │ │ Circuit Breakers │ │
│  │ Model Select   │  │ │ │ Reranker     │ │ │ │ A/B Testing      │ │
│  │ Provider Call  │  │ │ └──────────────┘ │ │ └──────────────────┘ │
│  └────────────────┘  │ └──────────────────┘ └──────────────────────┘
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INTELLIGENCE LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   NSGA-II    │  │   Bandits    │  │   Online ML  │  │   Judges     │    │
│  │   Weights    │  │   (UCB1/TS)  │  │   (River)    │  │   (LLM)      │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROVIDER LAYER                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Ollama     │  │   OpenAI     │  │   Anthropic  │  │   Gemini     │    │
│  │   (Local)    │  │   (Cloud)    │  │   (Cloud)    │  │   (Cloud)    │    │
│  │              │  │              │  │              │  │              │    │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │    │
│  │ │ Circuit  │ │  │ │ Circuit  │ │  │ │ Circuit  │ │  │ │ Circuit  │ │    │
│  │ │ Breaker  │ │  │ │ Breaker  │ │  │ │ Breaker  │ │  │ │ Breaker  │ │    │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │   MariaDB    │  │    Redis     │  │   ChromaDB   │                       │
│  │              │  │              │  │              │                       │
│  │ - Query Logs │  │ - Cache      │  │ - Vectors    │                       │
│  │ - EMA History│  │ - Bandits    │  │ - RAG Docs   │                       │
│  │ - Settings   │  │ - Settings   │  │ - Sem. Cache │                       │
│  │ - Judges     │  │ - Rate Limit │  │              │                       │
│  └──────────────┘  └──────────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Query Flow

### Sequence Diagram

```
┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐
│Client│     │ API  │     │Router│     │Bandit│     │Prov. │     │Celery│
└──┬───┘     └──┬───┘     └──┬───┘     └──┬───┘     └──┬───┘     └──┬───┘
   │            │            │            │            │            │
   │ POST /query│            │            │            │            │
   │───────────>│            │            │            │            │
   │            │            │            │            │            │
   │            │ Middleware │            │            │            │
   │            │ (rate lim, │            │            │            │
   │            │ backpress) │            │            │            │
   │            │────────────│            │            │            │
   │            │            │            │            │            │
   │            │ route_and_ │            │            │            │
   │            │ answer()   │            │            │            │
   │            │───────────>│            │            │            │
   │            │            │            │            │            │
   │            │            │ Check      │            │            │
   │            │            │ Semantic   │            │            │
   │            │            │ Cache      │            │            │
   │            │            │────────────│            │            │
   │            │            │            │            │            │
   │            │            │ Compute UQ │            │            │
   │            │            │────────────│            │            │
   │            │            │            │            │            │
   │            │            │ Get Top-2  │            │            │
   │            │            │ Candidates │            │            │
   │            │            │────────────│            │            │
   │            │            │            │            │            │
   │            │            │ select_    │            │            │
   │            │            │ model()    │            │            │
   │            │            │───────────>│            │            │
   │            │            │            │            │            │
   │            │            │ chosen     │            │            │
   │            │            │ model      │            │            │
   │            │            │<───────────│            │            │
   │            │            │            │            │            │
   │            │            │ RAG Aug.   │            │            │
   │            │            │ (optional) │            │            │
   │            │            │────────────│            │            │
   │            │            │            │            │            │
   │            │            │ call_model │            │            │
   │            │            │ ()         │            │            │
   │            │            │───────────────────────>│            │
   │            │            │            │            │            │
   │            │            │ response + │            │            │
   │            │            │ metadata   │            │            │
   │            │            │<───────────────────────│            │
   │            │            │            │            │            │
   │            │ response   │            │            │            │
   │            │<───────────│            │            │            │
   │            │            │            │            │            │
   │ 200 OK     │            │            │            │            │
   │<───────────│            │            │            │            │
   │            │            │            │            │            │
   │            │ Dispatch   │            │            │            │
   │            │ background │            │            │            │
   │            │ task       │            │            │            │
   │            │──────────────────────────────────────────────────>│
   │            │            │            │            │            │
```

### Detailed Steps

1. **Request Ingress**
   - GZip decompression
   - Rate limiting check
   - Backpressure check (503 if at capacity)
   - Correlation ID assignment

2. **Cache Lookup**
   - Semantic cache check using embedding similarity
   - Cache hit returns immediately (0 cost, ~0 latency)

3. **Uncertainty Quantification**
   - Generate query embedding
   - Compare against cluster centroids
   - Return uncertainty score (0-1)

4. **Candidate Selection**
   - Apply NSGA-II weights to filter candidates
   - Hard constraints (modality, availability)
   - Return top-2 models for bandit

5. **Bandit Selection**
   - Meta-bandit selects algorithm (ε-greedy, UCB1, Thompson)
   - Final model selection with exploration

6. **RAG Augmentation** (if enabled)
   - Vector similarity search
   - Optional reranking
   - Prompt augmentation

7. **Provider Call**
   - Connection pooling
   - Circuit breaker check
   - Retry with exponential backoff
   - Adaptive timeout

8. **Response & Metrics**
   - Cost calculation
   - Metrics recording
   - Response formatting

9. **Background Feedback** (Celery)
   - Quality judgment (sampled)
   - Bandit update
   - EMA update
   - Online ML update
   - Cache storage

---

## Feedback Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    BACKGROUND FEEDBACK LOOP                      │
│                    (Celery Worker)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. SAMPLING DECISION                                            │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │ monte_carlo_prob = 1 / sqrt(n_samples)                   │ │
│     │ pred_error_prob = online_predictor.predict(embedding)    │ │
│     │ final_prob = max(monte_carlo_prob, pred_error_prob)      │ │
│     │ if model in SOTA: final_prob *= 0.1                      │ │
│     │ final_prob = max(JUDGE_MIN_SAMPLE_RATE, final_prob)      │ │
│     └─────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌────────────────────────┐      ┌────────────────────────┐
│  2a. JUDGE EVALUATION  │      │  2b. SKIP JUDGMENT     │
│  (if sampled)          │      │  (use historical avg)  │
│                        │      │                        │
│  - Select 1-3 judges   │      │  quality = model_ema   │
│  - Evaluate response   │      │                        │
│  - Compute consensus   │      │                        │
└───────────┬────────────┘      └───────────┬────────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. REWARD COMPUTATION                                           │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │ reward = compute_reward(model, quality, latency, cost)   │ │
│     │        = weighted combination normalized to [0, 1]       │ │
│     └─────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. STATE UPDATES                                                │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Bandit State │  │ EMA History  │  │ Online ML    │          │
│  │ (Redis)      │  │ (MariaDB)    │  │ (Disk)       │          │
│  │              │  │              │  │              │          │
│  │ Update Q(a)  │  │ Update EMAs  │  │ Learn from   │          │
│  │ Update N(a)  │  │ for quality, │  │ actual       │          │
│  │              │  │ latency,cost │  │ outcome      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. CACHE UPDATE                                                 │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │ if quality >= 7.0:                                       │ │
│     │     store_cache(query, response, embedding)              │ │
│     └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Data Stores

```
┌─────────────────────────────────────────────────────────────────┐
│                         MariaDB                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ settings_dynamic     │ Dynamic configuration key-values    │  │
│  │ query_log            │ Query history with metrics          │  │
│  │ ema_history          │ Current EMA values per model        │  │
│  │ ema_history_log      │ Historical EMA snapshots            │  │
│  │ judge_calibration    │ Judge score calibration data        │  │
│  │ judge_performance_log│ Judge performance windows           │  │
│  │ user_feedback        │ Explicit user feedback              │  │
│  │ ab_experiments       │ A/B test definitions                │  │
│  │ ab_results           │ A/B test outcomes                   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          Redis                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ settings:*           │ Settings cache (30s TTL)            │  │
│  │ bandit:state:*       │ Bandit algorithm state              │  │
│  │ ema_latency:*        │ EMA latency cache (60s TTL)         │  │
│  │ rate_limit:*         │ Rate limit counters                 │  │
│  │ semantic_cache:*     │ Semantic cache entries              │  │
│  │ health_check         │ Health check cache (30s TTL)        │  │
│  │ settings:reload      │ PubSub channel for hot-reload       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         ChromaDB                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ text_collection      │ Text document embeddings            │  │
│  │ image_collection     │ Image embeddings (CLIP)             │  │
│  │ multimodal_collection│ Multimodal embeddings               │  │
│  │ cache_collection     │ Semantic cache embeddings           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration Cascade

```
┌─────────────┐
│ Environment │ ◄── Highest priority
│ Variables   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Redis    │ ◄── Fast access, 30s TTL
│    Cache    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   MariaDB   │ ◄── Persistent, admin-modifiable
│   Table     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Defaults   │ ◄── Code-defined fallbacks
│  (Python)   │
└─────────────┘
```

---

## Service Architecture

### Docker Compose Services

| Service | Port | Role | Dependencies |
|---------|------|------|--------------|
| **api** | 8000 | FastAPI application | mariadb, redis, ollama |
| **celery-worker** | - | Background task processor | mariadb, redis |
| **mariadb** | 3307 | Relational database | - |
| **redis** | 6378 | Cache & message broker | - |
| **ollama** | 11434 | Local LLM inference | - |
| **chromadb** | 8001 | Vector database | - |
| **prometheus** | 9090 | Metrics collection | api |
| **grafana** | 3000 | Dashboards | prometheus |
| **loki** | 3100 | Log aggregation | promtail |
| **promtail** | - | Log shipper | api |
| **locust** | 8089 | Load testing | api |

### Scaling Considerations

```
┌─────────────────────────────────────────────────────────────────┐
│                    HORIZONTAL SCALING                            │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │  API #1  │  │  API #2  │  │  API #N  │  ◄── Stateless        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                       │
│       │             │             │                              │
│       └─────────────┼─────────────┘                              │
│                     │                                            │
│              ┌──────┴──────┐                                     │
│              │ Load        │                                     │
│              │ Balancer    │                                     │
│              └─────────────┘                                     │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │ Celery#1 │  │ Celery#2 │  │ Celery#N │  ◄── Parallel workers │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                       │
│       │             │             │                              │
│       └─────────────┼─────────────┘                              │
│                     │                                            │
│              ┌──────┴──────┐                                     │
│              │   Redis     │  ◄── Shared state                   │
│              │   Cluster   │                                     │
│              └─────────────┘                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Resilience Patterns

| Pattern | Implementation | Location |
|---------|---------------|----------|
| **Circuit Breaker** | pybreaker | `providers_async.py` |
| **Retry + Backoff** | tenacity | `providers_async.py` |
| **Rate Limiting** | Redis-backed | `middleware/rate_limit.py` |
| **Backpressure** | Semaphore | `middleware/backpressure.py` |
| **Request Timeout** | asyncio.wait_for | `router_core.py` |
| **Request Dedup** | In-flight tracking | `reliability.py` |
| **Graceful Shutdown** | FastAPI lifecycle | `main.py` |
| **Health Checks** | Deep + Cached | `health.py` |

---

## Monitoring Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                      OBSERVABILITY                               │
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   API       │────>│ Prometheus  │────>│  Grafana    │       │
│  │  /metrics   │     │             │     │ Dashboards  │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   API       │────>│  Promtail   │────>│    Loki     │       │
│  │   Logs      │     │             │     │             │       │
│  └─────────────┘     └─────────────┘     └──────┬──────┘       │
│                                                  │               │
│                                                  ▼               │
│                                          ┌─────────────┐        │
│                                          │  Grafana    │        │
│                                          │  Logs View  │        │
│                                          └─────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Key Metrics

- `api_requests_total`: Total API requests
- `api_latency_seconds`: Request latency histogram
- `router_chosen_total{model}`: Model selection counts
- `bandit_reward`: Reward distribution
- `circuit_breaker_state{model}`: Circuit breaker status
- `backpressure_rejected_total`: Rejected requests
- `cache_hit_rate`: Semantic cache effectiveness
