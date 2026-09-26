# Objective: Refuse to serve traffic against a schema the code does not expect.
"""Verify at startup that the database has the columns this build writes.

The API validated its ``ADMIN_TOKEN`` and its critical settings at startup — it
failed fast on a bad configuration — and then happily served traffic against a
half-migrated database. Nothing compared ``alembic current`` with ``head``, and
``ensure_runtime_support_tables`` is a no-op in production.

That mattered because ``db_init`` cannot report a failed migration: its compose
command joins the steps with ``;`` and ends with an ``echo``, so the container
exits 0 whatever alembic did, and ``service_completed_successfully`` is always
satisfied. The only remaining place to catch it is here.

Comparing revisions would require alembic at runtime and a writable version
table. Checking the *columns the code actually writes* is both cheaper and a
closer question: a migration that ran but left a column behind is exactly as
broken as one that never ran.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

from sqlalchemy import text

logger = logging.getLogger(__name__)

#: Colunas que esta versão do código escreve e sem as quais o INSERT falha em
#: runtime — na primeira requisição real, e em silêncio, porque `persist_log`
#: apanha a excepção.
REQUIRED_COLUMNS = {
    "query_log": (
        "quality_semantics",
        "q_tech",
        "q_calibrado",
        "p_entrega",
        "detected_complexity",
        "decision_json",
        "correlation_id",
        "participant",
        "episode_id",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "finish_reason",
        "trace_json",
    ),
    "judge_logs": ("delivery_level", "correlation_id"),
    "request_failures": ("correlation_id", "status_code", "category"),
}


def missing_columns(engine, required=None) -> List[str]:
    """``["table.column", ...]`` for everything this build needs and cannot find.

    A table that does not exist yet is not reported: it is created by
    ``db_manager`` on first boot, and a fresh install would otherwise be
    indistinguishable from a stale one.
    """
    required = required or REQUIRED_COLUMNS
    missing: List[str] = []
    with engine.connect() as conn:
        for table, columns in required.items():
            if not _table_exists(conn, table):
                continue
            present = _columns_of(conn, table)
            missing.extend(f"{table}.{c}" for c in columns if c not in present)
    return missing


def _table_exists(conn, table: str) -> bool:
    result = conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = :t"
        ),
        {"t": table},
    )
    return bool(result.scalar())


def _columns_of(conn, table: str) -> set:
    result = conn.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = :t"
        ),
        {"t": table},
    )
    return {row[0] for row in result}


def verify_schema(*, strict: bool) -> Iterable[str]:
    """Log — and in strict mode refuse — a schema missing expected columns.

    ``strict`` is the operator's call on which failure is worse: serving with a
    stale schema, where rows are silently written without their new columns, or
    not serving at all. Production defaults to strict; development does not,
    because a developer mid-migration should not be locked out.
    """
    try:
        from app.db import get_engine

        missing = missing_columns(get_engine())
    except Exception as exc:
        # Uma base de dados inacessível é o problema do health check, não deste.
        logger.warning("[startup] Não foi possível verificar o esquema: %s", exc)
        return []

    if not missing:
        logger.info("[startup] Esquema verificado: todas as colunas esperadas existem.")
        return []

    message = (
        f"Esquema desatualizado: faltam {len(missing)} coluna(s) — {', '.join(sorted(missing))}. "
        f"Corra `cd app && alembic upgrade head`."
    )
    if strict:
        raise RuntimeError(message)
    logger.error("[startup] %s", message)
    return missing
