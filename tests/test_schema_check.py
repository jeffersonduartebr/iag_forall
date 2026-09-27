# Objective: The API must refuse to serve against a schema it cannot write to.
"""Nothing compared the running schema with what the code writes.

Startup validated the ``ADMIN_TOKEN`` and the critical settings — it failed
fast on a bad configuration — and then served traffic against a half-migrated
database. The failure surfaced only as missing ``query_log`` rows, because
``persist_log`` caught the insert error.

This is the last place it can be caught: ``db_init`` cannot report a failed
migration at all, since its compose command joins the steps with ``;`` and ends
with an ``echo``, so the container exits 0 whatever alembic did.
"""

import pytest
from app.services.schema_check import REQUIRED_COLUMNS, missing_columns, verify_schema


class FakeConn:
    """``information_schema.columns`` of the current schema, from ``{table: {columns}}``."""

    def __init__(self, columns):
        self.columns = columns
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, statement, params=None):
        self.statements.append(str(statement))
        return iter([(t, c) for t, cols in self.columns.items() for c in cols])


class FakeEngine:
    def __init__(self, columns):
        self._conn = FakeConn(columns)

    def connect(self):
        return self._conn


def _complete():
    return {t: set(c) for t, c in REQUIRED_COLUMNS.items()}


def test_a_fully_migrated_database_reports_nothing():
    assert missing_columns(FakeEngine(_complete())) == []


def test_the_lookup_is_scoped_to_the_current_postgresql_schema():
    engine = FakeEngine(_complete())
    missing_columns(engine)
    assert len(engine._conn.statements) == 1
    assert "current_schema()" in engine._conn.statements[0]
    assert "DATABASE()" not in engine._conn.statements[0]


def test_a_stale_database_names_every_missing_column():
    columns = _complete()
    columns["query_log"].discard("decision_json")
    columns["query_log"].discard("correlation_id")

    missing = missing_columns(FakeEngine(columns))
    assert set(missing) == {"query_log.decision_json", "query_log.correlation_id"}


def test_a_table_that_does_not_exist_is_a_failure_too():
    """Nothing creates tables at runtime any more (alembic_pg owns the schema): a missing table is a missing
    migration, reported column by column."""
    columns = _complete()
    del columns["request_failures"]
    missing = missing_columns(FakeEngine(columns))
    assert missing and all(m.startswith("request_failures.") for m in missing)
    assert "request_failures.status_code" in missing


def test_strict_mode_refuses_to_start(monkeypatch):
    columns = _complete()
    columns["query_log"].discard("decision_json")
    monkeypatch.setattr("app.db.get_engine", lambda: FakeEngine(columns))

    with pytest.raises(RuntimeError, match="Esquema desatualizado"):
        verify_schema(strict=True)


def test_non_strict_mode_logs_and_continues(monkeypatch):
    """A developer mid-migration should not be locked out of their own stack."""
    columns = _complete()
    columns["query_log"].discard("decision_json")
    monkeypatch.setattr("app.db.get_engine", lambda: FakeEngine(columns))

    assert list(verify_schema(strict=False)) == ["query_log.decision_json"]


def test_an_unreachable_database_is_not_this_checks_problem(monkeypatch):
    """That is what the health check is for; failing here would turn a
    transient outage into a refusal to boot."""

    def _boom():
        raise RuntimeError("mariadb em baixo")

    monkeypatch.setattr("app.db.get_engine", _boom)
    assert list(verify_schema(strict=True)) == []


def test_the_required_columns_are_the_ones_the_insert_writes():
    """Every column the INSERT writes is checked at startup (a column the INSERT writes and the schema lacks is a
    failed insert on the first real request)."""
    import inspect
    import re

    from app.query_service import insert_query_log

    source = inspect.getsource(insert_query_log)
    match = re.search(r"INSERT INTO query_log\s*\((.*?)\)\s*VALUES", source, re.S)
    written = {c.strip() for c in match.group(1).replace("\n", " ").split(",") if c.strip()}
    assert written <= set(REQUIRED_COLUMNS["query_log"])
