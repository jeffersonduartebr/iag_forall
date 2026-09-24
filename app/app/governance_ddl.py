# Objective: Deploy step that creates the governance tables (budgets, policies, RBAC, evals) strictly.
"""``python -m app.governance_ddl`` (run by ``db_init``).

In production the runtime DDL is forbidden (``ROADMAP_AUTO_DDL=0``) and no Alembic revision creates these
tables, so without this step a fresh production database never had them and every budget read failed.
Unlike ``roadmap_features.ensure_roadmap_tables`` (which only logs), a failure here fails the deploy.
"""

from __future__ import annotations

from sqlalchemy import text

from .db import get_engine
from .roadmap_features import DDL_STATEMENTS


def criar_tabelas_de_governanca() -> None:
    with get_engine().begin() as conn:
        for ddl in DDL_STATEMENTS:
            conn.execute(text(ddl))
        conn.execute(text("ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS metadata_json LONGTEXT NULL"))


if __name__ == "__main__":
    criar_tabelas_de_governanca()
    print("✅ Tabelas de governança prontas.")
