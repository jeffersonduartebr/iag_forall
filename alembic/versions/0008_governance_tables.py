# Objective: Alembic migration creating the governance tables (budgets, policies, RBAC, evals).
"""Governance tables as a migration, not a deploy-time side step

With ``ROADMAP_AUTO_DDL=0`` (production) the runtime DDL is forbidden, and no revision created these tables:
a fresh production database never had them and every budget read failed (Caso 1 load test, 2026-09-24).
The first fix was an extra ``python -m app.governance_ddl`` step in ``db_init``; this revision replaces it,
so the schema has one owner again.

The statements are frozen here (a migration must not change when ``roadmap_features`` does). All are
``IF NOT EXISTS``: a database where the deploy step already created them upgrades as a no-op.

Revision ID: 0008_governance_tables
Revises: 0007_decision_audit
Create Date: 2026-09-25
"""

import sqlalchemy as sa
from alembic import op

revision = "0008_governance_tables"
down_revision = "0007_decision_audit"
branch_labels = None
depends_on = None

DDL = [
    """
    CREATE TABLE IF NOT EXISTS tenant_budgets (
        tenant_id VARCHAR(128) PRIMARY KEY,
        daily_usd_limit FLOAT NOT NULL DEFAULT 0,
        monthly_usd_limit FLOAT NOT NULL DEFAULT 0,
        enabled TINYINT NOT NULL DEFAULT 1,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS tenant_usage (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        tenant_id VARCHAR(128) NOT NULL,
        day_key DATE NOT NULL,
        month_key VARCHAR(7) NOT NULL,
        requests INT NOT NULL DEFAULT 0,
        tokens_in BIGINT NOT NULL DEFAULT 0,
        tokens_out BIGINT NOT NULL DEFAULT 0,
        cost_usd FLOAT NOT NULL DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_tenant_day (tenant_id, day_key),
        INDEX idx_tenant_month (tenant_id, month_key)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        actor VARCHAR(128) NOT NULL,
        action VARCHAR(128) NOT NULL,
        resource VARCHAR(128) NOT NULL,
        tenant_id VARCHAR(128) NULL,
        metadata LONGTEXT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_action_created (action, created_at),
        INDEX idx_tenant_created (tenant_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS policy_versions (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        version VARCHAR(128) NOT NULL UNIQUE,
        description TEXT,
        config_json LONGTEXT,
        is_active TINYINT NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_active (is_active)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_runs (
        id VARCHAR(64) PRIMARY KEY,
        status VARCHAR(32) NOT NULL,
        policy_version VARCHAR(128) NULL,
        tenant_id VARCHAR(128) NULL,
        notes TEXT,
        prompts_json LONGTEXT,
        summary_json LONGTEXT,
        metadata_json LONGTEXT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_status_created (status, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_run_results (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        run_id VARCHAR(64) NOT NULL,
        prompt_text TEXT,
        model VARCHAR(255),
        quality FLOAT,
        latency_s FLOAT,
        cost_usd FLOAT,
        metadata_json LONGTEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_run_created (run_id, created_at),
        INDEX idx_run_model (run_id, model)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS rbac_user_roles (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        user_id VARCHAR(128) NOT NULL,
        role_name VARCHAR(64) NOT NULL,
        tenant_id VARCHAR(128) NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_user_role_tenant (user_id, role_name, tenant_id),
        INDEX idx_user (user_id),
        INDEX idx_role (role_name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS response_reviews (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        correlation_id VARCHAR(128) NULL,
        tenant_id VARCHAR(128) NULL,
        query_text TEXT,
        answer LONGTEXT,
        chosen_model VARCHAR(255),
        confidence_score FLOAT NULL,
        confidence_band VARCHAR(16) NULL,
        grounded TINYINT DEFAULT 0,
        verification_status VARCHAR(32) NULL,
        review_status VARCHAR(32) NOT NULL DEFAULT 'needs_review',
        review_reason VARCHAR(64) NULL,
        reviewer_id VARCHAR(128) NULL,
        reviewer_notes LONGTEXT NULL,
        corrected_answer LONGTEXT NULL,
        metadata_json LONGTEXT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_review_status_created (review_status, created_at),
        INDEX idx_tenant_review_created (tenant_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS expert_profiles (
        user_id VARCHAR(128) PRIMARY KEY,
        display_name VARCHAR(255) NULL,
        theme_ids LONGTEXT NULL,
        credentials_note TEXT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS expert_accounts (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(255) NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        display_name VARCHAR(255) NOT NULL,
        phone VARCHAR(32) NULL,
        enabled TINYINT NOT NULL DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_expert_enabled (enabled, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS expert_assessments (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        expert_id VARCHAR(128) NOT NULL,
        benchmark_id VARCHAR(128) NOT NULL,
        theme VARCHAR(128) NOT NULL,
        query_text TEXT,
        answer LONGTEXT,
        reference LONGTEXT NULL,
        eval_run_id VARCHAR(64) NULL,
        judge_quality FLOAT NULL,
        quality_score FLOAT NOT NULL,
        rubric_json LONGTEXT NULL,
        notes LONGTEXT NULL,
        status VARCHAR(32) NOT NULL DEFAULT 'submitted',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_expert_benchmark_run (expert_id, benchmark_id, eval_run_id),
        INDEX idx_theme_created (theme, created_at),
        INDEX idx_eval_run (eval_run_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
]

TABLES = [
    "tenant_budgets",
    "tenant_usage",
    "audit_log",
    "policy_versions",
    "eval_runs",
    "eval_run_results",
    "rbac_user_roles",
    "response_reviews",
    "expert_profiles",
    "expert_accounts",
    "expert_assessments",
]


def upgrade() -> None:
    conn = op.get_bind()
    for ddl in DDL:
        conn.execute(sa.text(ddl))
    conn.execute(sa.text("ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS metadata_json LONGTEXT NULL"))


def downgrade() -> None:
    conn = op.get_bind()
    for table in reversed(TABLES):
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {table}"))
