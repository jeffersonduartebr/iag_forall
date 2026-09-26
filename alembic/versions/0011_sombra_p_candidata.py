# Objective: Alembic migration recording each shadow row's candidate inclusion probability.
"""Sombra enxuta: shadow_evaluations.p_candidata

Each sampled request now runs a uniform subset of the other candidates (``SHADOW_CANDIDATE_FRACTION``). The row of
every candidate, drawn or left out (``status = 'fora_da_amostra'``), records its inclusion probability ``k / N``;
the delivered answer's row records 1. Historical rows are NULL: they ran every candidate (probability 1).

Revision ID: 0011_sombra_p_candidata
Revises: 0010_registro_completo
Create Date: 2026-09-26
"""

import sqlalchemy as sa
from alembic import op

revision = "0011_sombra_p_candidata"
down_revision = "0010_registro_completo"
branch_labels = None
depends_on = None

TABELA, COLUNA = "shadow_evaluations", "p_candidata"


def upgrade() -> None:
    op.get_bind().execute(sa.text(f"ALTER TABLE {TABELA} ADD COLUMN IF NOT EXISTS {COLUNA} FLOAT NULL"))


def downgrade() -> None:
    op.get_bind().execute(sa.text(f"ALTER TABLE {TABELA} DROP COLUMN IF EXISTS {COLUNA}"))
