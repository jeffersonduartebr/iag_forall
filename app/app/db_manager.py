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
FALLBACK_HISTORY_JSON = os.path.join(DATA_DIR, "bandit_history_fallback.json")

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
  model_name    VARCHAR(128) NULL,
  model_family  VARCHAR(64) NULL,
  token_key     VARCHAR(64) NULL,
  PRIMARY KEY (id),
  KEY idx_created (created_at),
  KEY idx_family_created (model_family, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

# ✅ DDL CORRIGIDO: usa model_family como chave primária
DDL_NSGA_CURRENT = """
CREATE TABLE IF NOT EXISTS nsga_current_weights (
  model_family  VARCHAR(64) NOT NULL,
  updated_at   DATETIME(6) NOT NULL,
  w_q          DOUBLE NOT NULL,
  w_c          DOUBLE NOT NULL,
  w_l          DOUBLE NOT NULL,
  fitness_mean DOUBLE NOT NULL,
  generations  INT NOT NULL,
  model_name   VARCHAR(128) NULL,
  token_key    VARCHAR(64) NULL,
  PRIMARY KEY (model_family)
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
    """
    last_err: Optional[BaseException] = None
    
    # ✅ Inicializa 'conn' como None fora do loop
    # Isso garante que a variável esteja sempre vinculada
    conn: Optional[pymysql.connections.Connection] = None

    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            conn = _connect()
            _ensure_schema(conn)
            conn.ping(reconnect=True)
            yield conn
            return # Sai com sucesso
        except Exception as e:
            last_err = e
            logger.warning(f"[db] connection attempt {attempt}/{DB_MAX_RETRIES} failed: {e}")
            time.sleep(DB_RETRY_DELAY_S)
        finally:
            # Esta verificação agora é segura, pois 'conn'
            # está vinculado a None ou a um objeto de conexão.
            try:
                # ✅ Verificação simplificada e segura
                if conn and conn.open:
                    conn.close()
            except Exception:
                pass # Evita erro de fechamento
                
    raise RuntimeError(f"[db] unable to obtain DB connection: {last_err}")

# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def insert_history(model: str, reward: float, ema: float, query_sample: str = "") -> None:
    # ... (função síncrona permanece a mesma) ...
    if not model:
        logger.warning("[db] insert_history ignored: empty model")
        return

    sql = """
        INSERT INTO bandit_history (ts_utc, model, reward, ema, query_sample)
        VALUES (UTC_TIMESTAMP(6), %s, %s, %s, %s)
    """
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (model, float(reward), float(ema), query_sample[:250]))
                conn.commit()
                logger.debug(f"[db] Bandit history inserted: {model}, r={reward:.3f}, ema={ema:.3f}")
                return
        except Exception as e:
            logger.warning(f"[db] insert_history attempt {attempt} failed: {e}")
            time.sleep(DB_RETRY_DELAY_S)
    _append_fallback_history(model, reward, ema, query_sample)


def _append_fallback_history(model: str, reward: float, ema: float, query_sample: str = "") -> None:
    # ... (função síncrona permanece a mesma) ...
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "reward": float(reward),
        "ema": float(ema),
        "query_sample": query_sample[:250],
    }
    try:
        history = []
        if os.path.exists(FALLBACK_HISTORY_JSON):
            with open(FALLBACK_HISTORY_JSON, "r") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        history = data
                except Exception:
                    pass
        history.append(record)
        with open(FALLBACK_HISTORY_JSON, "w") as f:
            json.dump(history[-500:], f, indent=2)
        logger.warning(f"[db] fallback history updated: {FALLBACK_HISTORY_JSON}")
    except Exception as fe:
        logger.error(f"[db] failed to write fallback JSON: {fe}")


def load_history(limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Reads latest bandit history from DB.
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
                # ✅ CORREÇÃO: Especifique o DictCursor aqui
                with conn.cursor(DictCursor) as cur:
                    cur.execute(sql, (int(limit),))
                    rows = cur.fetchall()
                    return rows or []
        except Exception as e:
            logger.warning(f"[db] load_history attempt {attempt} failed: {e}")
            time.sleep(DB_RETRY_DELAY_S)
    logger.error("[db] load_history permanently failed; returning empty list.")
    return []


def insert_weights(theta: List[float], fitness_mean: float, generations: int,
                   model_name: str = "ollama/deepseek-r1:8b",
                   model_family: str = "deepseek",
                   token_key: str = "max_tokens") -> None:
    """
    Insere novos pesos NSGA-II no banco e atualiza a tabela 'nsga_current_weights'.
    Agora usa 'model_family' como chave para o upsert.
    """
    if not (isinstance(theta, (list, tuple)) and len(theta) == 3):
        raise ValueError("theta deve ser uma lista [w_q, w_c, w_l]")

    w_q, w_c, w_l = map(float, theta)

    insert_sql = """
        INSERT INTO nsga_weights (
            created_at, w_q, w_c, w_l, fitness_mean, generations,
            model_name, model_family, token_key
        )
        VALUES (UTC_TIMESTAMP(6), %s, %s, %s, %s, %s, %s, %s, %s)
    """

    # ✅ UPSERT CORRIGIDO: Usa model_family como chave
    upsert_sql = """
        INSERT INTO nsga_current_weights (
            model_family, updated_at, w_q, w_c, w_l, fitness_mean, generations,
            model_name, token_key
        )
        VALUES (%s, UTC_TIMESTAMP(6), %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
          updated_at=VALUES(updated_at),
          w_q=VALUES(w_q),
          w_c=VALUES(w_c),
          w_l=VALUES(w_l),
          fitness_mean=VALUES(fitness_mean),
          generations=VALUES(generations),
          model_name=VALUES(model_name),
          token_key=VALUES(token_key);
    """

    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            with get_conn() as conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(insert_sql, (
                            w_q, w_c, w_l, fitness_mean, generations,
                            model_name, model_family, token_key
                        ))
                        # ✅ Parâmetros corrigidos para o upsert
                        cur.execute(upsert_sql, (
                            model_family, # Chave Primária
                            w_q, w_c, w_l, fitness_mean, generations,
                            model_name, token_key
                        ))
                    conn.commit()
                    logger.info(
                        f"[db] Pesos NSGA persistidos θ=({w_q:.3f},{w_c:.3f},{w_l:.3f}) | "
                        f"fit={fitness_mean:.4f} | modelo={model_name} ({model_family}) | token={token_key}"
                    )
                    _write_fallback_json(theta, fitness_mean, generations)
                    return
                except Exception as tx_err:
                    conn.rollback()
                    raise tx_err
        except Exception as e:
            logger.warning(f"[db] insert_weights tentativa {attempt}/{DB_MAX_RETRIES} falhou: {e}")
            time.sleep(DB_RETRY_DELAY_S)

    logger.error("[db] insert_weights falhou após múltiplas tentativas; escrevendo fallback JSON.")
    _write_fallback_json(theta, fitness_mean, generations)


# ------------------------------------------------------------------------------
# ✅ NOVA FUNÇÃO: Ler pesos
# ------------------------------------------------------------------------------
def get_current_weights(model_family: str) -> Optional[List[float]]:
    """
    Carrega os pesos NSGA-II mais recentes para uma família de modelo específica.
    """
    sql = "SELECT w_q, w_c, w_l FROM nsga_current_weights WHERE model_family = %s LIMIT 1"
    try:
        with get_conn() as conn:
            # ✅ CORREÇÃO: Especifique o DictCursor aqui
            with conn.cursor(DictCursor) as cur:
                cur.execute(sql, (model_family,))
                row = cur.fetchone()
                if row:
                    weights = [float(row['w_q']), float(row['w_c']), float(row['w_l'])]
                    logger.debug(f"[db] Carregou pesos para '{model_family}': {weights}")
                    return weights
    except Exception as e:
        logger.warning(f"[db] get_current_weights falhou para '{model_family}': {e}")
        return None
    return None

# ------------------------------------------------------------------------------
# Fallback JSON helpers
# ------------------------------------------------------------------------------
def _write_fallback_json(theta: List[float], fitness_mean: float, generations: int) -> None:
    # ... (função síncrona permanece a mesma) ...
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