# app/db_manager.py
from __future__ import annotations

import os
import time
import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import pymysql
from pymysql.cursors import DictCursor

logger = logging.getLogger("db-manager")

# ------------------------------------------------------------------------------
# Environment / Defaults
# ------------------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "mariadb")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "router_user")
DB_PASS = os.getenv("DB_PASS", "router_pass")
DB_NAME = os.getenv("DB_NAME", "routerdb")
DB_CONN_TIMEOUT = int(os.getenv("DB_CONN_TIMEOUT", "8"))
DB_READ_TIMEOUT = int(os.getenv("DB_READ_TIMEOUT", "10"))
DB_WRITE_TIMEOUT = int(os.getenv("DB_WRITE_TIMEOUT", "10"))
DB_MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES", "5"))
DB_RETRY_DELAY_S = float(os.getenv("DB_RETRY_DELAY_S", "3.0"))

# Local fallback for last best weights if DB is down
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
os.makedirs(DATA_DIR, exist_ok=True)
FALLBACK_WEIGHTS_JSON = os.path.join(DATA_DIR, "weights.json")

# ------------------------------------------------------------------------------
# SQL (idempotent DDL)
# ------------------------------------------------------------------------------
DDL_BANDIT_HISTORY = """
CREATE TABLE IF NOT EXISTS bandit_history (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  ts_utc       DATETIME(6) NOT NULL,
  model        VARCHAR(128) NOT NULL,
  reward       DOUBLE NOT NULL,
  ema          DOUBLE NOT NULL,
  query_sample VARCHAR(256) NULL,
  PRIMARY KEY (id),
  KEY idx_ts (ts_utc),
  KEY idx_model_ts (model, ts_utc)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

DDL_NSGA_WEIGHTS = """
CREATE TABLE IF NOT EXISTS nsga_weights (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  created_at    DATETIME(6) NOT NULL,
  w_q           DOUBLE NOT NULL,
  w_c           DOUBLE NOT NULL,
  w_l           DOUBLE NOT NULL,
  fitness_mean  DOUBLE NOT NULL,
  generations   INT NOT NULL,
  PRIMARY KEY (id),
  KEY idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

DDL_NSGA_CURRENT = """
CREATE TABLE IF NOT EXISTS nsga_current_weights (
  id           TINYINT NOT NULL,
  updated_at   DATETIME(6) NOT NULL,
  w_q          DOUBLE NOT NULL,
  w_c          DOUBLE NOT NULL,
  w_l          DOUBLE NOT NULL,
  fitness_mean DOUBLE NOT NULL,
  generations  INT NOT NULL,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

# ------------------------------------------------------------------------------
# Connection helpers
# ------------------------------------------------------------------------------
def _connect() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        autocommit=False,
        connect_timeout=DB_CONN_TIMEOUT,
        read_timeout=DB_READ_TIMEOUT,
        write_timeout=DB_WRITE_TIMEOUT,
        charset="utf8mb4",
        cursorclass=DictCursor,
    )

def _ensure_schema(conn: pymysql.connections.Connection) -> None:
    """Ensure tables exist (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(DDL_BANDIT_HISTORY)
        cur.execute(DDL_NSGA_WEIGHTS)
        cur.execute(DDL_NSGA_CURRENT)
    conn.commit()

@contextmanager
def get_conn():
    """
    Context manager with retry + ping-before-use to keep connections healthy.
    Usage:
        with get_conn() as conn:
            ...
    """
    last_err: Optional[BaseException] = None
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            conn = _connect()
            # Ensure schema once per connection lifecycle
            _ensure_schema(conn)
            # ping to revalidate connection if pooled/reused
            conn.ping(reconnect=True)
            yield conn
            # on normal exit, commit is caller’s responsibility.
            return
        except Exception as e:
            last_err = e
            logger.warning(
                f"[db] connection attempt {attempt}/{DB_MAX_RETRIES} failed: {e}"
            )
            time.sleep(DB_RETRY_DELAY_S)
        finally:
            try:
                if "conn" in locals() and conn and conn.open:
                    conn.close()
            except Exception:
                pass
    # Exhausted retries
    raise RuntimeError(f"[db] unable to obtain DB connection: {last_err}")

# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def load_history(limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Reads latest bandit history from DB.
    Returns a list of dicts with: ts_utc, model, reward, ema, query_sample
    """
    sql = """
        SELECT ts_utc, model, reward, ema, query_sample
        FROM bandit_history
        ORDER BY ts_utc DESC
        LIMIT %s
    """
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (int(limit),))
                    rows = cur.fetchall()
                    # caller expects recent→old order; already DESC
                    return rows or []
        except Exception as e:
            logger.warning(f"[db] load_history attempt {attempt} failed: {e}")
            time.sleep(DB_RETRY_DELAY_S)
    logger.error("[db] load_history permanently failed; returning empty list.")
    return []

def insert_weights(theta: List[float], fitness_mean: float, generations: int) -> None:
    """
    Transactionally inserts a new weights row into nsga_weights AND
    upserts nsga_current_weights (id=1) as the latest “current best”.
    On failure, writes a fallback JSON for the app to read.
    """
    if not (isinstance(theta, (list, tuple)) and len(theta) == 3):
        raise ValueError("theta must be a 3-vector [w_q, w_c, w_l]")

    w_q, w_c, w_l = float(theta[0]), float(theta[1]), float(theta[2])

    insert_sql = """
        INSERT INTO nsga_weights (created_at, w_q, w_c, w_l, fitness_mean, generations)
        VALUES (UTC_TIMESTAMP(6), %s, %s, %s, %s, %s)
    """
    upsert_sql = """
        INSERT INTO nsga_current_weights (id, updated_at, w_q, w_c, w_l, fitness_mean, generations)
        VALUES (1, UTC_TIMESTAMP(6), %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
          updated_at=VALUES(updated_at),
          w_q=VALUES(w_q),
          w_c=VALUES(w_c),
          w_l=VALUES(w_l),
          fitness_mean=VALUES(fitness_mean),
          generations=VALUES(generations);
    """

    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            with get_conn() as conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(insert_sql, (w_q, w_c, w_l, float(fitness_mean), int(generations)))
                        cur.execute(upsert_sql, (w_q, w_c, w_l, float(fitness_mean), int(generations)))
                    conn.commit()
                    logger.info(
                        f"[db] weights persisted θ=({w_q:.3f},{w_c:.3f},{w_l:.3f}) "
                        f"fit={fitness_mean:.4f} gen={generations}"
                    )
                    _write_fallback_json(theta, fitness_mean, generations)
                    return
                except Exception as tx_err:
                    conn.rollback()
                    raise tx_err
        except Exception as e:
            logger.warning(
                f"[db] insert_weights attempt {attempt}/{DB_MAX_RETRIES} failed: {e}"
            )
            time.sleep(DB_RETRY_DELAY_S)

    # Hard failure: write fallback file
    logger.error("[db] insert_weights failed after retries; writing fallback JSON.")
    _write_fallback_json(theta, fitness_mean, generations)

# ------------------------------------------------------------------------------
# Fallback JSON helpers (used by app if DB is down)
# ------------------------------------------------------------------------------

def _write_fallback_json(theta: List[float], fitness_mean: float, generations: int) -> None:
    """
    Writes the “current_best” weights into the local JSON file
    for compatibility with components that still read weights.json.
    """
    payload = {
        "current_best": [float(theta[0]), float(theta[1]), float(theta[2])],
        "fitness_mean": float(fitness_mean),
        "generations": int(generations),
        "updated_at": _utc_now_iso(),
        "source": "db_fallback"
    }
    try:
        with open(FALLBACK_WEIGHTS_JSON, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"[db] fallback weights.json updated at {FALLBACK_WEIGHTS_JSON}")
    except Exception as e:
        logger.warning(f"[db] failed to write fallback JSON: {e}")

def _utc_now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
