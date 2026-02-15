# Repository Guidelines

## Project Structure & Module Organization
- Core API and routing logic live in `app/app/` (for example `main.py`, `router_core.py`, `providers_async.py`, `semantic_cache.py`).
- Automated tests live in `tests/` with unit, integration, smoke, and performance suites (`test_*.py`), plus load scenarios in `tests/locustfile.py`.
- Infrastructure files are at the repository root: `docker-compose.yml`, `alembic/`, `grafana/`, `prometheus/`, `loki/`, and `promtail/`.
- Runtime/state directories (`state/`, `data/`, `chromadb-data/`, `ollama_data/`) support local execution and should be treated as environment data.

## Build, Test, and Development Commands
- `docker compose up -d --build`: build and start the full local stack (API, DB, Redis, Ollama, observability).
- `docker compose logs -f api`: tail API logs.
- `cd app && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`: run API without Compose.
- `PYTHONPATH=app pytest -q tests`: run test suite from repo root.
- `PYTHONPATH=app pytest -q tests/test_smoke.py`: run smoke tests only.
- `ruff check app/app tests` and `ruff format app/app tests`: lint/format code.
- `mypy app/app`: static type checks for core modules.

## Coding Style & Naming Conventions
- Target runtime is Python 3.11.
- Use 4-space indentation, line length up to 120, and double quotes (configured in `pyproject.toml`).
- Keep imports sorted and grouped (`ruff` rule `I`).
- Follow PEP 8 naming: `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Name tests `test_<feature>.py` and test functions `test_<behavior>`.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio`, and `pytest-cov`.
- For async paths, use `@pytest.mark.asyncio`.
- Reuse `tests/conftest.py` mocks for Redis/DB/Chroma to avoid external dependency coupling.
- No enforced coverage threshold is defined; keep or improve coverage for touched files.

## Commit & Pull Request Guidelines
- This repository currently has no Git commit history; adopt Conventional Commits going forward (for example `feat: add fallback scoring`, `fix: handle redis timeout`).
- Keep commits focused and atomic.
- PRs should include: objective, key changes, test evidence (commands/results), config or migration impact, and screenshots for dashboard/UI updates when applicable.
- Link related issues/tasks and document any required rollout or rollback steps.

## Security & Configuration Tips
- Store secrets in `.env`; never commit real credentials or API keys.
- If schema changes are introduced, create and apply Alembic migrations (`alembic revision --autogenerate -m "..."`, `alembic upgrade head`).
