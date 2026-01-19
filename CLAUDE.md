# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Objective LLM Router System that intelligently orchestrates multiple language models (Ollama, OpenAI, Gemini, Claude) while optimizing for three competing objectives: **cost**, **latency**, and **quality**. Combines NSGA-II multi-objective optimization, adaptive bandits (epsilon-greedy, UCB1, Thompson Sampling), and uncertainty quantification.

## Common Commands

### Running the Application
```bash
# Full stack with Docker Compose
docker-compose up -d

# API only (development)
cd app && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing
```bash
# All tests
pytest tests/

# Single test file
pytest tests/test_router_strategy.py -v

# Specific test
pytest tests/test_judges.py::test_judge_answer -v

# With coverage
pytest tests/ --cov=app/app --cov-report=html
```

### Code Quality
```bash
# Lint check
ruff check app/ tests/

# Auto-fix lint issues
ruff check --fix app/ tests/

# Format code
ruff format app/ tests/

# Type checking
mypy app/ --ignore-missing-imports

# All pre-commit hooks
pre-commit run --all-files
```

### Load Testing
```bash
# Via Docker
docker-compose up locust
# Access at http://localhost:8089

# Or directly
locust -f tests/locustfile.py --web-host 0.0.0.0
```

### Database Migrations
```bash
cd app && alembic upgrade head
```

## Architecture

### Core Flow
```
POST /query → router_core.py → bandits.py (model selection)
                            → router_strategy.py (NSGA-II scoring + filters)
                            → providers_async.py (LLM call)
                            → judges.py (quality assessment)
```

### Key Components

| Module | Purpose |
|--------|---------|
| `router_core.py` | Main query routing logic, EMA tracking, cache management |
| `bandits.py` | Meta-bandit system with multiple algorithms |
| `router_strategy.py` | Hard filters + NSGA-II weight application |
| `judges.py` | LLM-based quality assessment with consensus |
| `nsga_weights_updater.py` | Background NSGA-II optimization (DEAP) |
| `settings_dynamic.py` | 3-tier config (env → Redis → MariaDB) |
| `providers_async.py` | Async LLM provider abstraction |
| `rag_local.py` | Multimodal RAG (text + vision) |

### Background Services
- **NSGA-II Updater**: Periodic weight optimization (updates Redis)
- **Celery Worker**: Async feedback/judge processing
- **Meta Optimizer**: Bayesian hyperparameter tuning

### Data Layer
- **MariaDB**: Query logs, EMA history, settings, judge verdicts
- **Redis**: Bandit state, settings cache, semantic cache
- **ChromaDB**: Vector embeddings for RAG

## API Endpoints

```
POST /query              # Main query routing (accepts text, images, RAG flags)
GET  /health             # Health check
GET  /metrics            # Prometheus metrics
GET  /admin/settings     # Get current settings (requires admin token)
PUT  /admin/settings     # Update settings (requires admin token)
```

## Configuration

Settings are loaded in 3 layers (from `settings_dynamic.py`):
1. Environment variables (.env)
2. Redis LRU cache (30s TTL)
3. MariaDB persistent table

Key settings: `NSGA_W_QUALITY`, `NSGA_W_LATENCY`, `NSGA_W_COST`, `BANDIT_EPSILON`, `UNCERTAINTY_THRESHOLD`, `CANDIDATE_MODELS_LIST`, `JUDGE_LLMS`

## Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| api | 8000 | FastAPI main application |
| mariadb | 3307 | Database |
| redis | 6379 | Cache + state |
| ollama | 11434 | Local LLM inference |
| prometheus | 9090 | Metrics |
| grafana | 3000 | Dashboards |
| locust | 8089 | Load testing |
| loki | 3100 | Log aggregation |

## Code Style

- Python 3.11
- Line length: 120 (pyproject.toml)
- Ruff for linting/formatting
- MyPy for type checking (strict_optional, warn_redundant_casts)
- Pre-commit hooks enforced
