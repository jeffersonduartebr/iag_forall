# -*- coding: utf-8 -*-
"""
db.py — Centralized Database Connection Management
---------------------------------------------------
Provides a single database engine instance for the entire application,
preventing duplicate connection pools and improving resource management.

Usage:
    from app.db import get_engine, get_db_url

All modules should import from here instead of creating their own engines.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

# ==============================================================================
# Database Configuration
# ==============================================================================

def _get_db_config() -> dict:
    """Get database configuration from environment variables."""
    return {
        "host": os.getenv("DB_HOST", "mariadb"),
        "port": int(os.getenv("DB_PORT", "3306")),
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

    return (
        f"mysql+pymysql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )


# ==============================================================================
# Singleton Engine Instance
# ==============================================================================

_engine: Optional[Engine] = None
_engine_initialized: bool = False


def get_engine() -> Engine:
    """
    Get the singleton database engine instance.

    Creates the engine on first call with optimal pool settings:
    - pool_size=20: Maximum connections in the pool
    - max_overflow=10: Extra connections when pool is exhausted
    - pool_recycle=300: Recycle connections every 5 minutes to avoid stale connections
    - pool_pre_ping=True: Verify connections before use

    Returns:
        SQLAlchemy Engine instance.
    """
    global _engine, _engine_initialized

    if _engine is not None:
        return _engine

    if _engine_initialized:
        # Engine was initialized but is None (failed)
        raise RuntimeError("Database engine initialization failed previously")

    _engine_initialized = True

    try:
        db_url = get_db_url()

        _engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_pre_ping=True,
            pool_recycle=300,  # 5 minutes - prevents stale connections in cloud environments
            pool_size=20,      # Per-process baseline to avoid DB exhaustion with multiple workers
            max_overflow=10,   # Supports burst without overcommitting mariadb max_connections
            pool_timeout=60,   # Wait up to 60s for a connection
            echo=False,        # Set to True for SQL debugging
        )

        # Test the connection
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        logger.info(
            f"[db] Engine created: pool_size=20, max_overflow=10, "
            f"pool_recycle=300s, host={_get_db_config()['host']}"
        )

        return _engine

    except Exception as e:
        logger.error(f"[db] Failed to create database engine: {e}")
        _engine = None
        raise


def close_engine() -> None:
    """
    Close the database engine and dispose of all connections.

    Should be called during application shutdown.
    """
    global _engine, _engine_initialized

    if _engine is not None:
        try:
            _engine.dispose()
            logger.info("[db] Engine disposed, all connections closed")
        except Exception as e:
            logger.warning(f"[db] Error disposing engine: {e}")
        finally:
            _engine = None

    _engine_initialized = False


# ==============================================================================
# Connection Pool Statistics
# ==============================================================================

def get_pool_stats() -> dict:
    """
    Get database connection pool statistics.

    Returns:
        Dict with pool status information.
    """
    if _engine is None:
        return {
            "size": 0,
            "checked_in": 0,
            "checked_out": 0,
            "overflow": 0,
            "invalid": 0,
            "status": "not_initialized",
        }

    try:
        pool = _engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": getattr(pool, 'invalidatedcount', lambda: 0)(),
            "status": "healthy",
        }
    except Exception as e:
        logger.warning(f"[db] Error getting pool stats: {e}")
        return {
            "size": 0,
            "checked_in": 0,
            "checked_out": 0,
            "overflow": 0,
            "invalid": 0,
            "status": f"error: {e}",
        }


# ==============================================================================
# Health Check
# ==============================================================================

def check_db_health() -> dict:
    """
    Check database health and return detailed status.

    Returns:
        Dict with health status, latency, and pool info.
    """
    import time

    result = {
        "healthy": False,
        "latency_ms": None,
        "pool_stats": None,
        "error": None,
    }

    try:
        engine = get_engine()

        start = time.time()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency_ms = (time.time() - start) * 1000

        result["healthy"] = True
        result["latency_ms"] = round(latency_ms, 2)
        result["pool_stats"] = get_pool_stats()

    except Exception as e:
        result["error"] = str(e)

    return result


# ==============================================================================
# Backward Compatibility - Expose engine directly for existing code
# ==============================================================================

# For modules that import `engine` directly, we provide a lazy property
class _LazyEngine:
    """Lazy engine accessor for backward compatibility."""

    def __getattr__(self, name):
        """Resumo do comportamento desta função.

        Args:
            name: Parâmetro de entrada.

        Returns:
            Valor retornado pela função.
        """
        return getattr(get_engine(), name)

    def __call__(self, *args, **kwargs):
        """Resumo do comportamento desta função.

        Args:
            *args: Parâmetro de entrada.
            **kwargs: Parâmetro de entrada.

        Returns:
            Valor retornado pela função.
        """
        return get_engine()


# This allows: `from app.db import engine` to work
engine = _LazyEngine()
