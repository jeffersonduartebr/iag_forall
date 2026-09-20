# Objective: Tests that migration 0006 matches the bootstrap schema and is safe to re-run.
"""A migration and the bootstrap DDL that disagree produce two different databases.

``app.db_manager`` creates the schema on a fresh install; alembic upgrades an
existing one. If they drift, a column exists on new deployments and not on old
ones, and the code that writes it fails only in production. Every column this
migration adds is therefore checked against the bootstrap definition.

The second property is that the migration guards each step: it is run on
databases in unknown states, including ones where the application's own
``ADD COLUMN IF NOT EXISTS`` self-healing already added the columns.
"""

import importlib.util
from pathlib import Path

import pytest

from app import db_manager

MIGRATION = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "0006_formative_semantics.py"


@pytest.fixture(scope="module")
def migration():
    spec = importlib.util.spec_from_file_location("migration_0006", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


def test_the_migration_follows_the_previous_one(migration):
    assert migration.revision == "0006_formative_semantics"
    assert migration.down_revision == "0005_judge_rubric_json"


def test_the_revision_chain_has_no_gaps():
    """Two migrations claiming the same parent would make the head ambiguous."""
    versions = MIGRATION.parent
    parents = []
    for path in versions.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("down_revision"):
                parents.append(line.split("=", 1)[1].strip())
    assert len(parents) == len(set(parents))


# ---------------------------------------------------------------------------
# Agreement with the bootstrap schema
# ---------------------------------------------------------------------------


def bootstrap_columns(table: str) -> dict:
    schema = db_manager.SCHEMA_DEFINITIONS[table]
    columns = dict(schema.get("columns", {}))
    for line in schema["ddl"].splitlines():
        stripped = line.strip().rstrip(",")
        if stripped and not stripped.upper().startswith(
            ("CREATE", "UNIQUE", "INDEX", "PRIMARY", ")", "--")
        ):
            columns.setdefault(stripped.split()[0], stripped)
    return columns


@pytest.mark.parametrize(
    "column", ["quality_semantics", "q_tech", "q_calibrado", "p_entrega", "detected_complexity"]
)
def test_query_log_columns_exist_in_the_bootstrap_schema(migration, column):
    assert column in migration.QUERY_LOG_COLUMNS
    assert column in bootstrap_columns("query_log")


def test_judge_logs_gains_the_delivery_level(migration):
    assert "delivery_level" in migration.JUDGE_LOGS_COLUMNS
    assert "delivery_level" in bootstrap_columns("judge_logs")


def test_both_ema_tables_gain_the_semantics_column():
    assert "semantics" in bootstrap_columns("ema_history")
    assert "semantics" in bootstrap_columns("ema_history_log")


def test_the_definitions_agree_between_migration_and_bootstrap(migration):
    """A NULL default on one side and NOT NULL on the other is a latent bug."""
    bootstrap = bootstrap_columns("query_log")
    for column, definition in migration.QUERY_LOG_COLUMNS.items():
        assert bootstrap[column].upper().replace('"', "") .strip() == definition.upper().strip(), column


# ---------------------------------------------------------------------------
# The properties that make the columns safe
# ---------------------------------------------------------------------------


def test_the_semantics_column_is_never_null(migration):
    """A row whose semantics is unknown cannot be attributed to either meaning."""
    assert "NOT NULL" in migration.QUERY_LOG_COLUMNS["quality_semantics"]
    assert "rubric_v1" in migration.QUERY_LOG_COLUMNS["quality_semantics"]


def test_historical_rows_default_to_the_legacy_semantics(migration):
    assert "DEFAULT 'rubric_v1'" in migration.QUERY_LOG_COLUMNS["quality_semantics"]
    assert "DEFAULT 'rubric_v1'" in migration.EMA_SEMANTICS_COLUMN


@pytest.mark.parametrize("column", ["q_tech", "q_calibrado", "p_entrega", "detected_complexity"])
def test_the_new_measurements_are_nullable(migration, column):
    """NULL is the truth for a row measured before these existed."""
    assert "NULL" in migration.QUERY_LOG_COLUMNS[column]
    assert "NOT NULL" not in migration.QUERY_LOG_COLUMNS[column]


def test_the_ema_uniqueness_moves_to_include_the_semantics():
    """Otherwise the calibrated EMAs would overwrite the rubric ones."""
    ddl = db_manager.SCHEMA_DEFINITIONS["ema_history"]["ddl"]
    assert "uniq_model_modality_semantics (model, modality, semantics)" in ddl
    assert "UNIQUE KEY uniq_model_modality (" not in ddl


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


class FakeConn:
    """Answers the information_schema probes the migration makes."""

    def __init__(self, existing_tables, existing_columns, existing_indexes):
        self.tables = existing_tables
        self.columns = existing_columns
        self.indexes = existing_indexes

    def execute(self, statement, params=None):
        sql = str(statement)
        params = params or {}
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


def run_upgrade(migration, monkeypatch, conn):
    executed = []
    monkeypatch.setattr(migration.op, "get_bind", lambda: conn)
    monkeypatch.setattr(migration.op, "execute", lambda sql: executed.append(str(sql)))
    migration.upgrade()
    return executed


def test_a_fresh_database_gets_every_column(migration, monkeypatch):
    conn = FakeConn({"query_log", "judge_logs", "ema_history", "ema_history_log"}, set(), set())
    executed = run_upgrade(migration, monkeypatch, conn)
    joined = "\n".join(executed)
    for column in list(migration.QUERY_LOG_COLUMNS) + ["delivery_level", "semantics"]:
        assert column in joined, column


def test_re_running_the_migration_changes_nothing(migration, monkeypatch):
    """The application's own self-healing may have added the columns already."""
    tables = {"query_log", "judge_logs", "ema_history", "ema_history_log"}
    columns = {("query_log", c) for c in migration.QUERY_LOG_COLUMNS}
    columns |= {("judge_logs", "delivery_level")}
    columns |= {("ema_history", "semantics"), ("ema_history_log", "semantics")}
    indexes = {
        ("ema_history", "uniq_model_modality_semantics"),
        ("query_log", "idx_quality_semantics"),
    }
    executed = run_upgrade(migration, monkeypatch, FakeConn(tables, columns, indexes))
    assert executed == []


def test_a_database_without_the_tables_is_left_alone(migration, monkeypatch):
    """The tables are bootstrapped by db_manager; the migration only extends them."""
    executed = run_upgrade(migration, monkeypatch, FakeConn(set(), set(), set()))
    assert executed == []


def test_the_old_ema_index_is_replaced_not_duplicated(migration, monkeypatch):
    tables = {"ema_history"}
    columns = {("ema_history", "semantics")}
    indexes = {("ema_history", "uniq_model_modality")}
    executed = run_upgrade(migration, monkeypatch, FakeConn(tables, columns, indexes))
    joined = "\n".join(executed)
    assert "DROP INDEX uniq_model_modality" in joined
    assert "ADD UNIQUE KEY uniq_model_modality_semantics" in joined


def test_the_downgrade_removes_what_the_upgrade_added(migration, monkeypatch):
    tables = {"query_log", "judge_logs", "ema_history", "ema_history_log"}
    columns = {("query_log", c) for c in migration.QUERY_LOG_COLUMNS}
    columns |= {("judge_logs", "delivery_level")}
    columns |= {("ema_history", "semantics"), ("ema_history_log", "semantics")}
    indexes = {
        ("ema_history", "uniq_model_modality_semantics"),
        ("query_log", "idx_quality_semantics"),
    }
    executed = []
    monkeypatch.setattr(migration.op, "get_bind", lambda: FakeConn(tables, columns, indexes))
    monkeypatch.setattr(migration.op, "execute", lambda sql: executed.append(str(sql)))
    migration.downgrade()

    joined = "\n".join(executed)
    for column in list(migration.QUERY_LOG_COLUMNS) + ["delivery_level", "semantics"]:
        assert f"DROP COLUMN {column}" in joined, column
    assert "DROP INDEX idx_quality_semantics" in joined
