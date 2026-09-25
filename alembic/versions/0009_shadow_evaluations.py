# Objective: Alembic migration creating the research tables of the shadow execution (Caso 1).
"""Execução em sombra: shadow_evaluations and shadow_suspensions

One row per (sampled request, configuration): the delivered answer and every admissible candidate, with the judges'
scores, cost, latency, tokens and a SHA-256 of the text. No answer text is ever stored. Requests cut by the budget,
the per-tenant cap, the region or the GPU still get their rows (``executada = 0``), so the analysis can estimate the
effective inclusion probability per stratum and period. ``shadow_suspensions`` keeps the intervals in which the
shadow was suspended (budget, per-tenant cap).

These tables are research records, separate from query_log and from the OpenRouter exploration statistics.

Revision ID: 0009_shadow_evaluations
Revises: 0008_governance_tables
Create Date: 2026-09-25
"""

import sqlalchemy as sa
from alembic import op

revision = "0009_shadow_evaluations"
down_revision = "0008_governance_tables"
branch_labels = None
depends_on = None

DDL = [
    """
    CREATE TABLE IF NOT EXISTS shadow_evaluations (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        request_id VARCHAR(64) NOT NULL,
        episode_id VARCHAR(128) NULL,
        participante VARCHAR(256) NULL,
        tenant VARCHAR(128) NULL,
        caso TINYINT NULL,
        estrato LONGTEXT NULL,
        p_nominal DOUBLE NULL,
        executada TINYINT NOT NULL DEFAULT 0,
        motivo_corte VARCHAR(40) NULL,
        modelo VARCHAR(160) NOT NULL,
        provedor VARCHAR(40) NULL,
        regiao VARCHAR(40) NULL,
        versao_modelo VARCHAR(200) NULL,
        papel VARCHAR(10) NOT NULL,
        regime_entrega VARCHAR(20) NULL,
        p_atribuicao DOUBLE NULL,
        escores_juizes LONGTEXT NULL,
        escore_agregado DOUBLE NULL,
        painel LONGTEXT NULL,
        painel_uniforme TINYINT NULL,
        teria_abstido TINYINT NULL,
        custo_usd DOUBLE NULL,
        latencia_s DOUBLE NULL,
        tokens_entrada INT NULL,
        tokens_saida INT NULL,
        sha256_texto CHAR(64) NULL,
        status VARCHAR(40) NOT NULL,
        frozen_run_id VARCHAR(128) NULL,
        criado_em DATETIME NOT NULL,
        concluido_em DATETIME NULL,
        KEY ix_shadow_request (request_id),
        KEY ix_shadow_tenant_criado (tenant, criado_em),
        KEY ix_shadow_status (status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS shadow_suspensions (
        chave VARCHAR(200) PRIMARY KEY,
        motivo VARCHAR(40) NOT NULL,
        tenant VARCHAR(128) NULL,
        inicio DATETIME NOT NULL,
        fim DATETIME NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
]


def upgrade() -> None:
    conn = op.get_bind()
    for ddl in DDL:
        conn.execute(sa.text(ddl))


def downgrade() -> None:
    conn = op.get_bind()
    for table in ("shadow_suspensions", "shadow_evaluations"):
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {table}"))
