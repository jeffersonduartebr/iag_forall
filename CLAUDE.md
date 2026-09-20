# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Objective LLM Router System that intelligently orchestrates multiple language models (Ollama, OpenAI, Gemini, Claude) while optimizing for three competing objectives: **cost**, **latency**, and **quality**. Combines NSGA-II multi-objective optimization, adaptive bandits (epsilon-greedy, UCB1, Thompson Sampling), and uncertainty quantification.

## Common Commands

### Dependencies (uv)
The `.txt` files are the editable source; the `.lock` files are what CI and every
image actually install, with a sha256 per package (`--require-hashes`). A `.txt`
edited without regenerating its lock is silently ignored — the lock still wins.

```bash
# Install the dev environment from the lock
uv pip sync --require-hashes requirements-dev.lock

# After editing any requirements*.txt — regenerate and commit the lock
scripts/lock_requirements.sh              # all of them
scripts/lock_requirements.sh app/requirements.txt   # just one
```

| Lock | Source | Consumed by |
|---|---|---|
| `requirements-dev.lock` | `requirements-dev.txt` | CI (all 6 jobs) |
| `app/requirements.lock` | `app/requirements.txt` | `app/Dockerfile`, `Dockerfile.metaopt` |
| `app/requirements-db.lock` | `app/requirements-db.txt` | `app/Dockerfile.db-init` |
| `app/requirements.{correlation,nsga}.lock` | matching `.txt` | matching Dockerfile |

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
| redis | 6378 | Cache + state |
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
- Pre-commit hooks enforced (`.pre-commit-config.yaml`)
#### The quality gate (all three are required CI checks)

The three limits are one gate with three owners — each metric is enforced in
exactly one place, so a failure names a single cause.

| Limit | Owner | CI job |
|---|---|---|
| **SLOC < 200** per file (logical lines) | `scripts/check_file_length.py` | `lint_typecheck` |
| **CC < 15** per function | `ruff` C901 (`max-complexity = 14`) | `lint_typecheck` |
| **CRAP ≤ 30** per function | `scripts/crap_report.py` | `tests_unit` |

CRAP = CC² × (1 − cov)³ + CC, so the three compound: every extra branch costs
*squared* in risk and has to be paid for in coverage. CRAP lives in `tests_unit`
because it reads the `coverage.json` that job produces; splitting it out would
mean running the suite twice.

- **SLOC** is scoped to `app/app/` and `tests/`. The 44 files already over 200
  are grandfathered in `scripts/sloc_baseline.json` (ratchet: they may shrink,
  never grow) — **new files must be ≤200**. After shrinking a baselined file,
  lower its ceiling with `python3 scripts/check_file_length.py --update`.
  Refactor roadmap in `docs/SLOC_REFACTOR_ROADMAP.md`.
- **CRAP** has an empty baseline (`scripts/crap_baseline.json` is `{}`): nothing
  is grandfathered, so the limit is real for every function in the repo.
- Coverage (lines + branches) has its own CI floor (`--cov-fail-under=80`).
- Test tooling: fakeredis fixtures (`fake_redis`, `fake_aioredis`), `fake_clock` for TTLs
  (no real sleeps), hypothesis property tests (`tests/test_properties_*.py`), schemathesis
  contract tests (`pytest -m contract tests/contract`), benchmarks
  (`pytest tests/benchmarks -m benchmark --benchmark-enable`) and mutation testing
  (`scripts/run_mutation.sh`, config in `[tool.mutmut]`).
