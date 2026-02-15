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
DB_URL = (
    f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}"
    f"@{settings.DB_HOST}:3306/{settings.DB_NAME}"
)

target_metadata = None   # Não usamos autogenerate baseado em models.py

# -----------------------------------------------------------
# Modo offline
# -----------------------------------------------------------
def run_migrations_offline():
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
def run_migrations_online():
    engine = create_engine(DB_URL, poolclass=pool.NullPool)

    with engine.connect() as connection:
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
