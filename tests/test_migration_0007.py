# Objective: Migration 0007 must match the bootstrap schema and be safe to re-run.
"""A migration and the bootstrap DDL that disagree produce two databases.

``app.db_manager`` creates the schema on a fresh install; alembic upgrades an
existing one. If they drift, a column exists on new deployments and not on old
ones, and the code that writes it fails only in production.
"""

import importlib.util
from pathlib import Path

import pytest

from app import db_manager

MIGRATION = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "0007_decision_audit.py"


@pytest.fixture(scope="module")
def migration():
    spec = importlib.util.spec_from_file_location("migration_0007", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap_columns(table: str) -> dict:
    schema = db_manager.SCHEMA_DEFINITIONS[table]
    columns = dict(schema.get("columns", {}))
    for line in schema["ddl"].splitlines():
        stripped = line.strip().rstrip(",")
        if stripped and not stripped.upper().startswith(("CREATE", "UNIQUE", "INDEX", "PRIMARY", ")", "--")):
            columns.setdefault(stripped.split()[0], stripped)
    return columns


def test_the_migration_follows_the_previous_one(migration):
    assert migration.revision == "0007_decision_audit"
    assert migration.down_revision == "0006_formative_semantics"


def test_the_revision_chain_has_no_gaps():
    parents = []
    for path in MIGRATION.parent.glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("down_revision"):
                parents.append(line.split("=", 1)[1].strip())
    assert len(parents) == len(set(parents))


@pytest.mark.parametrize("column", ["decision_json", "correlation_id"])
def test_the_columns_exist_in_the_bootstrap_schema(migration, column):
    assert column in migration.QUERY_LOG_COLUMNS
    assert column in bootstrap_columns("query_log")


def test_both_columns_are_nullable(migration):
    """NULL is the truth for a row written before the decision was recorded.

    Inventing a decision for a historical row would be worse than the gap: the
    Thompson samples and the NSGA-II weights that produced it are gone.
    """
    for column, definition in migration.QUERY_LOG_COLUMNS.items():
        assert "NULL" in definition and "NOT NULL" not in definition, column


def test_the_self_healing_ddl_adds_the_same_columns():
    """The application adds columns on its own when alembic has not run."""
    import inspect

    from app import query_service

    source = inspect.getsource(query_service.ensure_query_log)
    assert "decision_json LONGTEXT NULL" in source
    assert "correlation_id VARCHAR(64) NULL" in source


class FakeConn:
    def __init__(self, tables, columns, indexes):
        self.tables, self.columns, self.indexes = tables, columns, indexes

    def execute(self, statement, params=None):
        sql, params = str(statement), params or {}
        if "information_schema.tables" in sql:
            return FakeResult(params["table"] in self.tables)
        if "information_schema.columns" in sql:
            return FakeResult((params["table"], params["column"]) in self.columns)
        if "information_schema.statistics" in sql:
            return FakeResult((params["table"], params["index"]) in self.indexes)
        raise AssertionError(f"consulta inesperada: {sql}")


class FakeResult:
    def __init__(self, value):
        self._value = 1 if value else 0

    def scalar(self):
        return self._value


def run(migration, monkeypatch, conn, direction="upgrade"):
    executed = []
    monkeypatch.setattr(migration.op, "get_bind", lambda: conn)
    monkeypatch.setattr(migration.op, "execute", lambda sql: executed.append(str(sql)))
    getattr(migration, direction)()
    return executed


def test_a_fresh_database_gets_both_columns_and_the_index(migration, monkeypatch):
    executed = run(migration, monkeypatch, FakeConn({"query_log"}, set(), set()))
    joined = "\n".join(executed)
    assert "decision_json" in joined and "correlation_id" in joined
    assert "CREATE INDEX idx_correlation_id" in joined


def test_re_running_the_migration_changes_nothing(migration, monkeypatch):
    """The application's own self-healing may have added the columns already."""
    conn = FakeConn(
        {"query_log"},
        {("query_log", c) for c in migration.QUERY_LOG_COLUMNS},
        {("query_log", "idx_correlation_id")},
    )
    assert run(migration, monkeypatch, conn) == []


def test_a_database_without_the_table_is_left_alone(migration, monkeypatch):
    assert run(migration, monkeypatch, FakeConn(set(), set(), set())) == []


def test_the_downgrade_removes_what_the_upgrade_added(migration, monkeypatch):
    conn = FakeConn(
        {"query_log"},
        {("query_log", c) for c in migration.QUERY_LOG_COLUMNS},
        {("query_log", "idx_correlation_id")},
    )
    joined = "\n".join(run(migration, monkeypatch, conn, direction="downgrade"))
    assert "DROP INDEX idx_correlation_id" in joined
    for column in migration.QUERY_LOG_COLUMNS:
        assert f"DROP COLUMN {column}" in joined, column
