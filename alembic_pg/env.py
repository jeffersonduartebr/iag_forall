"""Alembic environment of the PostgreSQL schema (Cloud SQL). The URL comes from ``app.db_url`` (DB_DIALECT etc.)."""

import os
import sys
from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context

# No container o pacote `app` está em /app (pai de alembic_pg/); no repositório, em ./app/app.
_PAI = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_PAI, "app") if os.path.isdir(os.path.join(_PAI, "app", "app")) else _PAI)

from app.db_url import dialeto, get_db_url  # noqa: E402

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

if dialeto() != "postgresql":
    raise RuntimeError("alembic_pg só roda com DB_DIALECT=postgresql")


def run_migrations_offline() -> None:
    context.configure(url=get_db_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(
        get_db_url(), poolclass=pool.NullPool, connect_args={"sslmode": os.getenv("DB_SSLMODE", "prefer")}
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None, transaction_per_migration=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
