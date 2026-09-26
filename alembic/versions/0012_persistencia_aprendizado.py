# Objective: Alembic migration making the router's learned state durable (EMA period + counts, learned_state).
"""Persistência do aprendizado: ema_history por período, learned_state

``ema_history`` gains ``policy_namespace`` (the study period of ``BANDIT_POLICY_NAMESPACE``; '' when unset, which
is every existing row) and ``updates`` (so a restored EMA keeps its weight in routing). Its uniqueness widens to
include the period: the new key is added first and the old one dropped after, so every existing row stays unique
throughout. ``learned_state`` is a durable key -> JSON copy of learned documents that used to live only in Redis
(semantic centroids, the exploration blocklist). ``bandit_context_stats`` moves its moments from FLOAT to DOUBLE.
The downgrade keeps DOUBLE: narrowing back would round the stored posteriors.

Revision ID: 0012_persistencia_aprendizado
Revises: 0011_sombra_p_candidata
Create Date: 2026-09-27
"""

import sqlalchemy as sa
from alembic import op

revision = "0012_persistencia_aprendizado"
down_revision = "0011_sombra_p_candidata"
branch_labels = None
depends_on = None

EMA_COLUNAS = {
    "policy_namespace": "VARCHAR(64) NOT NULL DEFAULT ''",
    "updates": "INT NOT NULL DEFAULT 0",
}
#: FLOAT (32 bits) acumulava erro no M2 de Welford a cada atualização; DOUBLE converte sem perda.
BANDIT_DOUBLE = {"avg_reward": "DOUBLE DEFAULT 0", "var": "DOUBLE DEFAULT 0", "M2": "DOUBLE DEFAULT 0"}
CHAVE_NOVA, CHAVE_VELHA = "uniq_ema_escopo", "uniq_model_modality_semantics"

LEARNED_STATE = """
CREATE TABLE IF NOT EXISTS learned_state (
    chave VARCHAR(191) PRIMARY KEY,
    valor LONGTEXT NOT NULL,
    atualizado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


def _existe(conn, sql: str, **params) -> bool:
    return bool(conn.execute(sa.text(sql), params).scalar())


def _tem_indice(conn, nome: str) -> bool:
    return _existe(conn, "SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema = DATABASE() "
                   "AND table_name = 'ema_history' AND index_name = :n", n=nome)


def upgrade() -> None:
    conn = op.get_bind()
    for nome, definicao in EMA_COLUNAS.items():
        conn.execute(sa.text(f"ALTER TABLE ema_history ADD COLUMN IF NOT EXISTS {nome} {definicao}"))
    if not _tem_indice(conn, CHAVE_NOVA):
        conn.execute(sa.text(
            f"ALTER TABLE ema_history ADD UNIQUE KEY {CHAVE_NOVA} (model, modality, semantics, policy_namespace)"
        ))
    if _tem_indice(conn, CHAVE_VELHA):
        conn.execute(sa.text(f"ALTER TABLE ema_history DROP INDEX {CHAVE_VELHA}"))
    conn.execute(sa.text(LEARNED_STATE))
    for nome, definicao in BANDIT_DOUBLE.items():
        conn.execute(sa.text(f"ALTER TABLE bandit_context_stats MODIFY COLUMN {nome} {definicao}"))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("DROP TABLE IF EXISTS learned_state"))
    # Voltar à chave estreita falha se houver linhas de mais de um período, e deve falhar: o downgrade não pode
    # apagar estado em silêncio. Apague as linhas dos outros períodos antes, se for mesmo a intenção.
    if not _tem_indice(conn, CHAVE_VELHA):
        conn.execute(sa.text(f"ALTER TABLE ema_history ADD UNIQUE KEY {CHAVE_VELHA} (model, modality, semantics)"))
    if _tem_indice(conn, CHAVE_NOVA):
        conn.execute(sa.text(f"ALTER TABLE ema_history DROP INDEX {CHAVE_NOVA}"))
    for nome in EMA_COLUNAS:
        conn.execute(sa.text(f"ALTER TABLE ema_history DROP COLUMN IF EXISTS {nome}"))
