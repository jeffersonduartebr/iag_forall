# Objective: Refuse to serve traffic against a schema the code does not expect.
"""Verify at startup that the database has the columns this build writes.

The API validated its ``ADMIN_TOKEN`` and its critical settings at startup — it
failed fast on a bad configuration — and then happily served traffic against a
half-migrated database. Nothing compared ``alembic current`` with ``head``, and
nothing creates tables at runtime (the schema belongs to ``alembic_pg``).

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

from .schema_colunas import REQUIRED_COLUMNS as _COLUNAS_DO_BASELINE

logger = logging.getLogger(__name__)

#: Colunas que esta versão do código escreve (todas as do baseline ``alembic_pg`` de cada tabela gravada) e sem
#: as quais o INSERT falha em runtime — na primeira requisição real, e em silêncio, porque `persist_log` apanha a
#: excepção.
REQUIRED_COLUMNS = _COLUNAS_DO_BASELINE


def missing_columns(engine, required=None) -> List[str]:
    """``["table.column", ...]`` for everything this build needs and cannot find.

    The schema has one owner, ``alembic -c alembic_pg.ini upgrade head``; nothing creates tables at runtime any
    more. A table that does not exist is therefore reported too (all its columns are missing): a fresh install
    that skipped the migration is exactly as broken as a stale one.
    """
    required = required or REQUIRED_COLUMNS
    with engine.connect() as conn:
        present = _present_columns(conn)
    return [f"{table}.{c}" for table, columns in required.items() for c in columns if (table, c) not in present]


def _present_columns(conn) -> set:
    """Every ``(table, column)`` of the current schema, in one query."""
    result = conn.execute(
        text("SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = current_schema()")
    )
    return {(row[0], row[1]) for row in result}


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
        f"Corra `alembic -c alembic_pg.ini upgrade head`."
    )
    if strict:
        raise RuntimeError(message)
    logger.error("[startup] %s", message)
    return missing
