# Objective: Alembic migration adding the formative quality fields and the EMA state namespace.
"""Formative evaluation: quality semantics, calibrated score and EMA namespace

Three things are added here, and one is not.

``query_log`` gains the formative fields beside ``quality``, which keeps its
meaning: the three-dimension rubric mean that the cache gate, the error
predictor, ROI and the golden sets all interpret. ``q_tech`` and ``q_calibrado``
sit next to it so the two can be compared on the same rows instead of one
replacing the other. Historical rows get ``quality_semantics = 'rubric_v1'`` and
NULL in the new columns, which is the truth rather than an invented value.

``ema_history`` gains a ``semantics`` column and its uniqueness moves to
``(model, modality, semantics)``. Without that, EMAs learned under the
calibrated score would overwrite the ones learned under the rubric mean, and the
two could never be compared or rolled back.

What is *not* here: ``bandit_context_stats`` needs no schema change. Its key is
``context_label``, a free-form string, so the namespace goes into the value and
old rows stay exactly where they are.

Revision ID: 0006_formative_semantics
Revises: 0005_judge_rubric_json
Create Date: 2026-09-20
"""

import sqlalchemy as sa
from alembic import op

revision = "0006_formative_semantics"
down_revision = "0005_judge_rubric_json"
branch_labels = None
depends_on = None


#: Column definitions added to each table, in the order they are applied.
QUERY_LOG_COLUMNS = {
    # Which meaning `quality` carries on this row. Never NULL: a row without it
    # would be unattributable, and mixing semantics silently is the whole risk.
    "quality_semantics": "VARCHAR(16) NOT NULL DEFAULT 'rubric_v1'",
    # Clarity and accuracy renormalised over their own weights.
    "q_tech": "FLOAT NULL",
    # Q_tech times the annihilation term.
    "q_calibrado": "FLOAT NULL",
    # Delivery factor in [0,1]; NULL when no usurpation judge answered.
    "p_entrega": "FLOAT NULL",
    # Complexity estimated before a model was chosen, for the routing matrix.
    "detected_complexity": "VARCHAR(16) NULL",
}

JUDGE_LOGS_COLUMNS = {
    # The ordinal level the judge picked, before conversion to p_entrega.
    "delivery_level": "FLOAT NULL",
}

EMA_SEMANTICS_COLUMN = "VARCHAR(16) NOT NULL DEFAULT 'rubric_v1'"


def _table_exists(conn, table_name: str) -> bool:
    result = conn.execute(
        sa.text(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = :table
            """
        ),
        {"table": table_name},
    )
    return bool(result.scalar())


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    result = conn.execute(
        sa.text(
            """
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = :table AND column_name = :column
            """
        ),
        {"table": table_name, "column": column_name},
    )
    return bool(result.scalar())


def _index_exists(conn, table_name: str, index_name: str) -> bool:
    result = conn.execute(
        sa.text(
            """
            SELECT COUNT(*) FROM information_schema.statistics
            WHERE table_schema = DATABASE() AND table_name = :table AND index_name = :index
            """
        ),
        {"table": table_name, "index": index_name},
    )
    return bool(result.scalar())


def _add_columns(conn, table: str, columns: dict) -> None:
    """Add each missing column; tables are bootstrapped by app.db_manager."""
    if not _table_exists(conn, table):
        return
    for name, definition in columns.items():
        if not _column_exists(conn, table, name):
            op.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def upgrade() -> None:
    conn = op.get_bind()

    _add_columns(conn, "query_log", QUERY_LOG_COLUMNS)
    _add_columns(conn, "judge_logs", JUDGE_LOGS_COLUMNS)
    _add_columns(conn, "ema_history", {"semantics": EMA_SEMANTICS_COLUMN})
    _add_columns(conn, "ema_history_log", {"semantics": EMA_SEMANTICS_COLUMN})

    # The uniqueness of an EMA row moves to include the semantics, otherwise the
    # calibrated EMAs would overwrite the rubric ones on the same (model, modality).
    if _table_exists(conn, "ema_history") and _column_exists(conn, "ema_history", "semantics"):
        if _index_exists(conn, "ema_history", "uniq_model_modality"):
            op.execute("ALTER TABLE ema_history DROP INDEX uniq_model_modality")
        if not _index_exists(conn, "ema_history", "uniq_model_modality_semantics"):
            op.execute(
                "ALTER TABLE ema_history "
                "ADD UNIQUE KEY uniq_model_modality_semantics (model, modality, semantics)"
            )

    # Analytical queries filter query_log by semantics before anything else.
    if _table_exists(conn, "query_log") and not _index_exists(conn, "query_log", "idx_quality_semantics"):
        op.execute("CREATE INDEX idx_quality_semantics ON query_log (quality_semantics)")


def downgrade() -> None:
    conn = op.get_bind()

    if _table_exists(conn, "query_log") and _index_exists(conn, "query_log", "idx_quality_semantics"):
        op.execute("DROP INDEX idx_quality_semantics ON query_log")

    if _table_exists(conn, "ema_history"):
        if _index_exists(conn, "ema_history", "uniq_model_modality_semantics"):
            op.execute("ALTER TABLE ema_history DROP INDEX uniq_model_modality_semantics")
        # Restoring the narrower key can fail if rows exist under more than one
        # semantics, which is the point: the downgrade must not silently drop
        # state. Delete the non-default rows first if that is genuinely intended.
        if not _index_exists(conn, "ema_history", "uniq_model_modality"):
            op.execute("ALTER TABLE ema_history ADD UNIQUE KEY uniq_model_modality (model, modality)")

    for table, columns in (
        ("ema_history_log", ["semantics"]),
        ("ema_history", ["semantics"]),
        ("judge_logs", list(JUDGE_LOGS_COLUMNS)),
        ("query_log", list(QUERY_LOG_COLUMNS)),
    ):
        if not _table_exists(conn, table):
            continue
        for column in columns:
            if _column_exists(conn, table, column):
                op.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
