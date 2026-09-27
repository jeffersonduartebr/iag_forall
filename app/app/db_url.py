# Objective: Build the database URL (MariaDB or PostgreSQL) from the environment, with no heavy imports.
"""The single place that knows the driver, the default port and how to quote credentials.

Kept free of SQLAlchemy so the settings bootstrap can build a fallback engine even when ``app.db`` cannot be
imported. ``app.db`` re-exports everything here.
"""

from __future__ import annotations

import os
from typing import Optional
from urllib.parse import quote_plus

#: Dialetos: PostgreSQL (Cloud SQL; o SQL da aplicação é PostgreSQL) e MariaDB (só histórico e ferramentas de cópia).
DIALETOS = {"mysql": ("mysql+pymysql", 3306), "postgresql": ("postgresql+psycopg2", 5432)}


def dialeto() -> str:
    """``DB_DIALECT`` (``postgresql`` default | ``mysql``); anything else is a configuration error."""
    valor = os.getenv("DB_DIALECT", "postgresql").strip().lower()
    if valor not in DIALETOS:
        raise ValueError(f"DB_DIALECT inválido: {valor!r} (use {', '.join(DIALETOS)})")
    return valor


def _get_db_config() -> dict:
    """Get database configuration from environment variables."""
    return {
        "dialect": dialeto(),
        "host": os.getenv("DB_HOST", "postgres"),
        "port": int(os.getenv("DB_PORT") or DIALETOS[dialeto()][1]),
        "user": os.getenv("DB_USER", "router_user"),
        "password": os.getenv("DB_PASS", ""),
        "database": os.getenv("DB_NAME", "routerdb"),
    }


def get_db_url(config: Optional[dict] = None) -> str:
    """
    Build the database URL from configuration.

    Args:
        config: Optional config dict. If None, uses environment variables.

    Returns:
        SQLAlchemy database URL string.
    """
    if config is None:
        config = _get_db_config()
    driver = DIALETOS[config.get("dialect", "postgresql")][0]
    senha = quote_plus(str(config["password"]))  # senha com @ : / não quebra a URL
    return f"{driver}://{config['user']}:{senha}@{config['host']}:{config['port']}/{config['database']}"
