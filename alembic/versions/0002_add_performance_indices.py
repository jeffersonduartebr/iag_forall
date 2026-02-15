"""Add performance indices for frequently queried tables

Revision ID: 0002_add_performance_indices
Revises: 0001_initial_multimodal_schema
Create Date: 2026-01-29

This migration adds indices to improve query performance:
- judge_calibration(created_at) - for time-based filtering
- ema_history_log(created_at) - for log retention queries
- judge_performance_log(window_end) - for window-based lookups
- query_log(created_at, chosen_model) - for analytics queries
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0002_add_performance_indices'
down_revision = '0001_initial_multimodal_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add performance indices."""
    conn = op.get_bind()

    def _table_exists(table_name: str) -> bool:
        """Executa table exists."""
        query = sa.text(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = :table_name
            LIMIT 1
            """
        )
        return conn.execute(query, {"table_name": table_name}).scalar() is not None

    def _index_exists(table_name: str, index_name: str) -> bool:
        """Executa index exists."""
        query = sa.text(
            """
            SELECT 1
            FROM information_schema.statistics
            WHERE table_schema = DATABASE()
              AND table_name = :table_name
              AND index_name = :index_name
            LIMIT 1
            """
        )
        return conn.execute(query, {"table_name": table_name, "index_name": index_name}).scalar() is not None

    def _create_index_if_needed(table_name: str, index_name: str, columns: str) -> None:
        """Executa create index if needed."""
        if _table_exists(table_name) and not _index_exists(table_name, index_name):
            op.execute(sa.text(f"CREATE INDEX {index_name} ON {table_name} ({columns})"))

    _create_index_if_needed("judge_calibration", "idx_judge_calibration_created", "created_at")
    _create_index_if_needed("ema_history_log", "idx_ema_history_log_created", "created_at")
    _create_index_if_needed("judge_performance_log", "idx_judge_perf_window_end", "window_end")
    _create_index_if_needed("query_log", "idx_query_log_created_model", "created_at, chosen_model")
    _create_index_if_needed("query_log", "idx_query_log_model", "chosen_model")
    _create_index_if_needed("ema_history", "idx_ema_history_model_modality", "model, modality")


def downgrade() -> None:
    """Remove performance indices."""
    conn = op.get_bind()

    def _table_exists(table_name: str) -> bool:
        """Executa table exists."""
        query = sa.text(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = :table_name
            LIMIT 1
            """
        )
        return conn.execute(query, {"table_name": table_name}).scalar() is not None

    def _index_exists(table_name: str, index_name: str) -> bool:
        """Executa index exists."""
        query = sa.text(
            """
            SELECT 1
            FROM information_schema.statistics
            WHERE table_schema = DATABASE()
              AND table_name = :table_name
              AND index_name = :index_name
            LIMIT 1
            """
        )
        return conn.execute(query, {"table_name": table_name, "index_name": index_name}).scalar() is not None

    def _drop_index_if_needed(table_name: str, index_name: str) -> None:
        """Executa drop index if needed."""
        if _table_exists(table_name) and _index_exists(table_name, index_name):
            op.execute(sa.text(f"DROP INDEX {index_name} ON {table_name}"))

    _drop_index_if_needed("judge_calibration", "idx_judge_calibration_created")
    _drop_index_if_needed("ema_history_log", "idx_ema_history_log_created")
    _drop_index_if_needed("judge_performance_log", "idx_judge_perf_window_end")
    _drop_index_if_needed("query_log", "idx_query_log_created_model")
    _drop_index_if_needed("query_log", "idx_query_log_model")
    _drop_index_if_needed("ema_history", "idx_ema_history_model_modality")
