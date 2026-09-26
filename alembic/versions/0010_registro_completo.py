# Objective: Alembic migration completing the per-request research record (Caso 1).
"""Registro completo: query_log research columns, judge_logs.correlation_id, request_failures

``query_log`` gains the participant (pseudonymous ``user_key``), the pedagogical episode, the token counts (reasoning
included), the finish reason and ``trace_json`` (stage timings, fallback chain, retrieval mode, policy/experiment,
the answer withheld by an abstention). ``judge_logs`` gains the correlation id, so each judge's score joins its
request. ``request_failures`` keeps the requests that ended in an error, which previously left no row anywhere.

Revision ID: 0010_registro_completo
Revises: 0009_shadow_evaluations
Create Date: 2026-09-26
"""

import sqlalchemy as sa
from alembic import op

revision = "0010_registro_completo"
down_revision = "0009_shadow_evaluations"
branch_labels = None
depends_on = None

QUERY_LOG_COLUMNS = {
    "participant": "VARCHAR(256) NULL",
    "episode_id": "VARCHAR(128) NULL",
    "prompt_tokens": "INT NULL",
    "completion_tokens": "INT NULL",
    "reasoning_tokens": "INT NULL",
    "finish_reason": "VARCHAR(32) NULL",
    "trace_json": "LONGTEXT NULL",
}

JUDGE_LOGS_COLUMNS = {
    "correlation_id": "VARCHAR(64) NULL",
}

DDL = [
    *(
        f"ALTER TABLE {tabela} ADD COLUMN IF NOT EXISTS {nome} {definicao}"
        for tabela, colunas in (("query_log", QUERY_LOG_COLUMNS), ("judge_logs", JUDGE_LOGS_COLUMNS))
        for nome, definicao in colunas.items()
    ),
    "CREATE INDEX IF NOT EXISTS ix_query_log_correlation ON query_log (correlation_id)",
    "CREATE INDEX IF NOT EXISTS ix_query_log_participant ON query_log (participant, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_judge_logs_correlation ON judge_logs (correlation_id)",
    """
    CREATE TABLE IF NOT EXISTS request_failures (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        correlation_id VARCHAR(64) NULL,
        tenant_id VARCHAR(128) NULL,
        participant VARCHAR(256) NULL,
        episode_id VARCHAR(128) NULL,
        route_path VARCHAR(64) NULL,
        status_code INT NOT NULL,
        category VARCHAR(64) NULL,
        model VARCHAR(255) NULL,
        detail_json TEXT NULL,
        query_text LONGTEXT NULL,
        modality VARCHAR(16) NULL,
        latency_s FLOAT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        KEY ix_request_failures_correlation (correlation_id),
        KEY ix_request_failures_tenant (tenant_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
]


def upgrade() -> None:
    conn = op.get_bind()
    for ddl in DDL:
        conn.execute(sa.text(ddl))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("DROP TABLE IF EXISTS request_failures"))
    for coluna in JUDGE_LOGS_COLUMNS:
        conn.execute(sa.text(f"ALTER TABLE judge_logs DROP COLUMN IF EXISTS {coluna}"))
    for coluna in QUERY_LOG_COLUMNS:
        conn.execute(sa.text(f"ALTER TABLE query_log DROP COLUMN IF EXISTS {coluna}"))
