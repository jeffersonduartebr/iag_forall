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
    def __init__(self, tables, columns):
        self.tables, self.columns = tables, columns

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, statement, params=None):
        sql, params = str(statement), params or {}
        if "information_schema.tables" in sql:
            return FakeResult([(1 if params["t"] in self.tables else 0,)])
        return FakeResult([(c,) for c in self.columns.get(params["t"], ())])


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalar(self):
        return self._rows[0][0]

    def __iter__(self):
        return iter(self._rows)


class FakeEngine:
    def __init__(self, tables, columns):
        self._conn = FakeConn(tables, columns)

    def connect(self):
        return self._conn


def test_a_fully_migrated_database_reports_nothing():
    engine = FakeEngine(set(REQUIRED_COLUMNS), {t: set(c) for t, c in REQUIRED_COLUMNS.items()})
    assert missing_columns(engine) == []


def test_a_stale_database_names_every_missing_column():
    columns = {t: set(c) for t, c in REQUIRED_COLUMNS.items()}
    columns["query_log"].discard("decision_json")
    columns["query_log"].discard("correlation_id")
    engine = FakeEngine(set(REQUIRED_COLUMNS), columns)

    missing = missing_columns(engine)
    assert set(missing) == {"query_log.decision_json", "query_log.correlation_id"}


def test_a_table_that_does_not_exist_yet_is_not_a_failure():
    """A fresh install would otherwise be indistinguishable from a stale one;
    db_manager creates the tables on first boot."""
    engine = FakeEngine(set(), {})
    assert missing_columns(engine) == []


def test_strict_mode_refuses_to_start(monkeypatch):
    columns = {t: set(c) for t, c in REQUIRED_COLUMNS.items()}
    columns["query_log"].discard("decision_json")
    monkeypatch.setattr("app.db.get_engine", lambda: FakeEngine(set(REQUIRED_COLUMNS), columns))

    with pytest.raises(RuntimeError, match="Esquema desatualizado"):
        verify_schema(strict=True)


def test_non_strict_mode_logs_and_continues(monkeypatch):
    """A developer mid-migration should not be locked out of their own stack."""
    columns = {t: set(c) for t, c in REQUIRED_COLUMNS.items()}
    columns["query_log"].discard("decision_json")
    monkeypatch.setattr("app.db.get_engine", lambda: FakeEngine(set(REQUIRED_COLUMNS), columns))

    assert list(verify_schema(strict=False)) == ["query_log.decision_json"]


def test_an_unreachable_database_is_not_this_checks_problem(monkeypatch):
    """That is what the health check is for; failing here would turn a
    transient outage into a refusal to boot."""

    def _boom():
        raise RuntimeError("mariadb em baixo")

    monkeypatch.setattr("app.db.get_engine", _boom)
    assert list(verify_schema(strict=True)) == []


def test_the_required_columns_are_the_ones_the_insert_writes():
    """A column added to the INSERT and forgotten here is a silent NULL."""
    import inspect
    import re

    from app.query_service import insert_query_log

    source = inspect.getsource(insert_query_log)
    match = re.search(r"INSERT INTO query_log\s*\((.*?)\)\s*VALUES", source, re.S)
    written = {c.strip() for c in match.group(1).replace("\n", " ").split(",") if c.strip()}
    assert set(REQUIRED_COLUMNS["query_log"]) <= written
