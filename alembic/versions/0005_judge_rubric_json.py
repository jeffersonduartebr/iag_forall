# Objective: Alembic migration for per-dimension judge rubric scores.
"""Add judge_logs.rubric_json for the three-dimension judge rubric

Revision ID: 0005_judge_rubric_json
Revises: 0004_openrouter_exploration_stats
Create Date: 2026-09-18
"""

from alembic import op
import sqlalchemy as sa


revision = "0005_judge_rubric_json"
down_revision = "0004_openrouter_exploration_stats"
branch_labels = None
depends_on = None


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


def upgrade() -> None:
    conn = op.get_bind()
    # judge_logs is bootstrapped by app.db_manager; only add the column when missing.
    if _table_exists(conn, "judge_logs") and not _column_exists(conn, "judge_logs", "rubric_json"):
        op.execute("ALTER TABLE judge_logs ADD COLUMN rubric_json TEXT NULL")


def downgrade() -> None:
    conn = op.get_bind()
    if _table_exists(conn, "judge_logs") and _column_exists(conn, "judge_logs", "rubric_json"):
        op.execute("ALTER TABLE judge_logs DROP COLUMN rubric_json")
