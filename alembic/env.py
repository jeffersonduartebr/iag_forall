# Objective: Alembic environment bootstrap for database migration execution.
"""Alembic environment bootstrap for database migration execution.

This module is part of the tracked codebase and should remain aligned with the
current runtime architecture and operational documentation.
"""


import sys
import os
from logging.config import fileConfig

from sqlalchemy import create_engine, pool
from alembic import context

# --- Importa settings_dynamic com caminho absoluto ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from app.settings_dynamic import settings

# -----------------------------------------------------------
# Carrega configurações de logging padrão
# -----------------------------------------------------------
config = context.config
fileConfig(config.config_file_name)

# -----------------------------------------------------------
# Construção da URL dinâmica do banco
# -----------------------------------------------------------
# A porta vem de DB_PORT: fixá-la em 3306 impedia o alembic de alcançar a base
# por qualquer porta publicada que não fosse a de dentro da rede do compose.
DB_URL = (
    f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}"
    f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
)

target_metadata = None   # Não usamos autogenerate baseado em models.py

# -----------------------------------------------------------
# Modo offline
# -----------------------------------------------------------
def run_migrations_offline():
    """Run migrations offline.

This function coordinates the main execution path for that step."""
    context.configure(
        url=DB_URL,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

# -----------------------------------------------------------
# Modo online
# -----------------------------------------------------------
#: Alembic creates ``alembic_version.version_num`` as VARCHAR(32) and offers no
#: way to configure it. This project's revision identifiers are descriptive and
#: some exceed that — ``0004_openrouter_exploration_stats`` is 33 characters —
#: so an upgrade fails mid-run with "Data too long for column", after earlier
#: revisions have already been applied. Widening the column up front makes the
#: chain runnable on a fresh database as well as on an existing one.
VERSION_TABLE_DDL = (
    "CREATE TABLE IF NOT EXISTS alembic_version ("
    " version_num VARCHAR(255) NOT NULL,"
    " CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
)
VERSION_COLUMN_WIDENING = "ALTER TABLE alembic_version MODIFY version_num VARCHAR(255) NOT NULL"


def ensure_version_table(connection) -> None:
    """Make sure the version table can hold this project's revision identifiers."""
    connection.exec_driver_sql(VERSION_TABLE_DDL)
    connection.exec_driver_sql(VERSION_COLUMN_WIDENING)


def run_migrations_online():
    """Run migrations online.

This function coordinates the main execution path for that step."""
    engine = create_engine(DB_URL, poolclass=pool.NullPool)

    with engine.connect() as connection:
        ensure_version_table(connection)
        connection.commit()
        context.configure(connection=connection)

        with context.begin_transaction():
            context.run_migrations()

# -----------------------------------------------------------
# Execução
# -----------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
