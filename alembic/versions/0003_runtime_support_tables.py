# Objective: Alembic migration for runtime support tables.
"""Add runtime support tables (user_feedback, model_metrics indexes)

Revision ID: 0003_runtime_support_tables
Revises: 0002_add_performance_indices
Create Date: 2026-07-04
"""

from alembic import op
import sqlalchemy as sa


revision = "0003_runtime_support_tables"
down_revision = "0002_add_performance_indices"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    def _table_exists(table_name: str) -> bool:
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

    if not _table_exists("user_feedback"):
        op.execute(
            """
            CREATE TABLE user_feedback (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                query_id VARCHAR(64),
                query_text TEXT,
                model VARCHAR(255) NOT NULL,
                modality VARCHAR(32),
                feedback_type VARCHAR(32) NOT NULL,
                rating INT,
                user_quality FLOAT NOT NULL,
                original_quality FLOAT,
                blended_quality FLOAT NOT NULL,
                reward FLOAT,
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_model (model),
                INDEX idx_feedback_type (feedback_type),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS user_feedback")
