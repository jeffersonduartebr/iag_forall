# Objective: Alembic migration for OpenRouter exploration stats persistence.
"""Add openrouter_exploration_stats table

Revision ID: 0004_openrouter_exploration_stats
Revises: 0003_runtime_support_tables
Create Date: 2026-07-04
"""

from alembic import op
import sqlalchemy as sa


revision = "0004_openrouter_exploration_stats"
down_revision = "0003_runtime_support_tables"
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

    if not _table_exists("openrouter_exploration_stats"):
        op.execute(
            """
            CREATE TABLE openrouter_exploration_stats (
                model VARCHAR(255) PRIMARY KEY,
                count INT NOT NULL DEFAULT 0,
                failure_count INT NOT NULL DEFAULT 0,
                mean_reward FLOAT,
                mean_latency_s FLOAT,
                mean_cost_usd FLOAT,
                mean_observed_usd_per_1k FLOAT,
                catalog_prompt_usd_per_1k FLOAT,
                catalog_completion_usd_per_1k FLOAT,
                auto_promoted_at TIMESTAMP NULL,
                blocklisted TINYINT NOT NULL DEFAULT 0,
                stats_json JSON,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_auto_promoted (auto_promoted_at),
                INDEX idx_blocklisted (blocklisted)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS openrouter_exploration_stats")
