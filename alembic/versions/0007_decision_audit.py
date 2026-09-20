# Objective: Alembic migration adding the routing-decision audit trail to query_log.
"""Decision audit: why a model was chosen, and which request it belonged to

``query_log`` recorded *which* model answered and never *why*. The response
already carried ``candidates`` and ``pareto_front``, but both were written as
literal empty lists in both routing paths, so the Pareto front that names the
central contribution of this work never reached disk. No amount of later
analysis can recover it: the Thompson samples, the shared EMAs and the NSGA-II
weights that produced a decision are all gone by the time the row is written.

``decision_json`` stores that record as it existed at decision time: every
candidate with its three objectives (sampled quality, estimated latency,
estimated cost), the risk multiplier applied to it, the scalarised score, a
flag for membership of the Pareto front, and the NSGA-II weights in force.
The weights matter as much as the scores — a background updater rewrites them,
so a score stored without them cannot be reinterpreted afterwards.

``correlation_id`` closes the other half: the identifier already threading the
structured logs never reached this table, so a row could not be joined to the
execution trace that produced it.

Both columns are NULL for historical rows, which is the truth. A row written
before this migration has no recoverable decision, and inventing one would be
worse than the gap.

Revision ID: 0007_decision_audit
Revises: 0006_formative_semantics
Create Date: 2026-09-20
"""

import sqlalchemy as sa
from alembic import op

revision = "0007_decision_audit"
down_revision = "0006_formative_semantics"
branch_labels = None
depends_on = None


QUERY_LOG_COLUMNS = {
    # JSON: candidatos com objectivos, frente de Pareto e pesos NSGA-II.
    # LONGTEXT e não JSON nativo, para acompanhar as outras colunas desta
    # tabela e não exigir uma versão mínima de MariaDB.
    "decision_json": "LONGTEXT NULL",
    # Liga a linha ao rasto de execução estruturado que a produziu.
    "correlation_id": "VARCHAR(64) NULL",
}


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


def upgrade() -> None:
    conn = op.get_bind()
    if not _table_exists(conn, "query_log"):
        return

    for name, definition in QUERY_LOG_COLUMNS.items():
        if not _column_exists(conn, "query_log", name):
            op.execute(f"ALTER TABLE query_log ADD COLUMN {name} {definition}")

    # Juntar uma linha ao seu rasto é a consulta que este campo existe para servir.
    if not _index_exists(conn, "query_log", "idx_correlation_id"):
        op.execute("CREATE INDEX idx_correlation_id ON query_log (correlation_id)")


def downgrade() -> None:
    conn = op.get_bind()
    if not _table_exists(conn, "query_log"):
        return

    if _index_exists(conn, "query_log", "idx_correlation_id"):
        op.execute("DROP INDEX idx_correlation_id ON query_log")

    for name in QUERY_LOG_COLUMNS:
        if _column_exists(conn, "query_log", name):
            op.execute(f"ALTER TABLE query_log DROP COLUMN {name}")
