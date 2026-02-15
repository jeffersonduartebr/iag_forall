"""
correlation_metrics.py — Cálculo, exposição e armazenamento histórico das correlações multiobjetivo (NSGA-II)
Autor: Jefferson Duarte

Funcionalidades:
  ✅ Calcula correlações dinâmicas entre latência, custo, qualidade e fitness.
  ✅ Expõe métricas Prometheus (para dashboards).
  ✅ Armazena o histórico no banco MariaDB para análises futuras e auditorias.

Requisitos de schema:
  - model_metrics(model, latency_ms, cost_usd, quality_score, fitness, generation, timestamp)
    (criado pelo init_db.sql)
  - correlation_history(id, model, corr_latency_quality, corr_cost_quality, corr_fitness_weights, r2_mean, generation, timestamp)
    (criado aqui, se não existir)

Execução:
  python app/app/correlation_metrics.py
"""

from __future__ import annotations

import os
import sys
import time
import math
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import redis
from datetime import datetime
from prometheus_client import start_http_server, Gauge
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError

# -----------------------------------------------------------------------------
# 🔧 Settings dinâmicos (Redis → DB → .env)
# -----------------------------------------------------------------------------
try:
    from app.settings_dynamic import settings
except ImportError:
    # Fallback de path se chamado de outro diretório
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from app.settings_dynamic import settings  # type: ignore

# -----------------------------------------------------------------------------
# 📝 Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("correlation-metrics")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# 🌍 Configurações (resolvidas via settings)
# -----------------------------------------------------------------------------
DB_HOST = settings.DB_HOST
DB_USER = settings.DB_USER
DB_PASS = settings.DB_PASS
DB_NAME = settings.DB_NAME

REDIS_HOST = settings.REDIS_HOST
REDIS_PORT = settings.REDIS_PORT
REDIS_PASS = settings.get("REDIS_PASS", "SenhaForte")

PROM_PORT = int(settings.get("CORR_PROM_PORT", 9105))
UPDATE_INTERVAL = int(settings.get("CORR_INTERVAL", 60))

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"

# -----------------------------------------------------------------------------
# 🔌 Conexões
# -----------------------------------------------------------------------------
def _make_db_engine() -> Any:
    """Cria engine SQLAlchemy com parâmetros seguros."""
    return create_engine(
        DB_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=5,
        max_overflow=5,
    )

def _connect_redis() -> redis.Redis | None:
    """Executa connect redis."""
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=int(REDIS_PORT),
            password=REDIS_PASS,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        # Teste rápido
        r.ping()
        return r
    except Exception as e:
        logger.warning(f"[redis] Falha ao conectar ao Redis ({REDIS_HOST}:{REDIS_PORT}): {e}")
        return None

db_engine = _make_db_engine()
redis_client = _connect_redis()

# -----------------------------------------------------------------------------
# ⏳ Espera resiliente por DB
# -----------------------------------------------------------------------------
def wait_for_db(max_wait_seconds: int = 120) -> None:
    """Espera o banco ficar disponível (com backoff exponencial)."""
    start = time.time()
    delay = 1.5
    attempt = 0
    while True:
        try:
            with db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ Conexão com MariaDB estabelecida.")
            return
        except OperationalError as e:
            elapsed = time.time() - start
            if elapsed > max_wait_seconds:
                logger.error(f"❌ Banco não respondeu após {max_wait_seconds}s: {e}")
                raise
            sleep_s = min(8.0, delay ** attempt)
            logger.info(f"⏳ Aguardando MariaDB... (tentativa {attempt+1}, dormindo {sleep_s:.1f}s)")
            time.sleep(sleep_s)
            attempt += 1

# -----------------------------------------------------------------------------
# 🗃️ Garantia do schema de histórico (idempotente)
# -----------------------------------------------------------------------------
def ensure_history_table() -> None:
    """Cria a tabela de histórico de correlação, se não existir."""
    ddl = """
    CREATE TABLE IF NOT EXISTS correlation_history (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        model VARCHAR(255),
        corr_latency_quality FLOAT,
        corr_cost_quality FLOAT,
        corr_fitness_weights FLOAT,
        r2_mean FLOAT,
        generation INT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_model_ts (model, timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    with db_engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("✅ Tabela 'correlation_history' verificada/criada.")

# -----------------------------------------------------------------------------
# 📥 Coleta de dados
# -----------------------------------------------------------------------------
def fetch_recent_metrics(window_sql: str = "1 DAY") -> pd.DataFrame:
    """
    Busca métricas recentes de 'model_metrics'.
    Requer o schema do init_db.sql (latency_ms, cost_usd, quality_score, fitness, generation, timestamp).
    """
    query = f"""
        SELECT model, latency_ms, cost_usd, quality_score, fitness, generation
        FROM model_metrics
        WHERE timestamp > NOW() - INTERVAL {window_sql};
    """
    try:
        df = pd.read_sql(query, db_engine)
        if df.empty:
            logger.warning("⚠️ Nenhum registro em 'model_metrics' na janela consultada.")
        return df
    except Exception as e:
        logger.warning(f"⚠️ Erro ao buscar 'model_metrics': {e}. Esta tabela existe?")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 🧮 Cálculo de correlações
# -----------------------------------------------------------------------------
def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlação de Pearson segura (retorna 0.0 se variância for zero ou input inválido)."""
    try:
        if len(a) < 2 or len(b) < 2:
            return 0.0
        if np.all(a == a[0]) or np.all(b == b[0]):
            return 0.0
        r = np.corrcoef(a, b)[0, 1]
        if math.isnan(r):
            return 0.0
        return float(r)
    except Exception:
        return 0.0

def compute_correlations(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Calcula correlações por modelo:
      - corr_lq: latência x qualidade
      - corr_cq: custo x qualidade
      - corr_fw: fitness x (latência + custo + qualidade) [proxy simples]
      - matrix: matriz 3x3 entre (lat, custo, qualidade)
      - generation: geração máxima observada no período
    """
    results: Dict[str, Dict[str, Any]] = {}

    if df.empty:
        return results

    # Assegura tipos numéricos
    for col in ("latency_ms", "cost_usd", "quality_score", "fitness", "generation"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latency_ms", "cost_usd", "quality_score", "fitness", "generation"])
    if df.empty:
        return results

    for model, group in df.groupby("model"):
        if len(group) < 3:
            continue

        lat = group["latency_ms"].to_numpy(dtype=float)
        cost = group["cost_usd"].to_numpy(dtype=float)
        qual = group["quality_score"].to_numpy(dtype=float)
        fit = group["fitness"].to_numpy(dtype=float)

        corr_lq = _safe_corr(lat, qual)
        corr_cq = _safe_corr(cost, qual)

        # Proxy simples de "peso combinado" para testar dependência com fitness
        combo = cost + lat + qual
        corr_fw = _safe_corr(fit, combo)

        # Matriz 3x3
        try:
            matrix = np.corrcoef(np.vstack([lat, cost, qual]))
        except Exception:
            matrix = np.eye(3)

        results[model] = {
            "corr_lq": corr_lq,
            "corr_cq": corr_cq,
            "corr_fw": corr_fw,
            "matrix": matrix,
            "labels": ["latency", "cost", "quality"],
            "generation": int(np.nanmax(group["generation"].to_numpy(dtype=float))),
        }

    return results

# -----------------------------------------------------------------------------
# 📈 Métricas Prometheus
# -----------------------------------------------------------------------------
correlation_latency_quality = Gauge(
    "correlation_latency_quality",
    "Correlação latência vs qualidade",
    ["model"],
)
correlation_cost_quality = Gauge(
    "correlation_cost_quality",
    "Correlação custo vs qualidade",
    ["model"],
)
correlation_fitness_weights = Gauge(
    "correlation_fitness_weights",
    "Correlação fitness vs combinação (lat+custo+qualidade)",
    ["model"],
)
nsga_correlation_r2_mean = Gauge(
    "nsga_correlation_r2_mean",
    "Coeficiente R² médio global",
)
model_correlation_matrix = Gauge(
    "model_correlation_matrix",
    "Matriz média de correlação (latência,custo,qualidade)",
    ["model", "metric_x", "metric_y"],
)

def publish_metrics(corr_data: Dict[str, Dict[str, Any]]) -> None:
    """Publica métricas Prometheus por modelo e o R² médio global."""
    all_r2 = []

    for model, data in corr_data.items():
        correlation_latency_quality.labels(model=model).set(data["corr_lq"])
        correlation_cost_quality.labels(model=model).set(data["corr_cq"])
        correlation_fitness_weights.labels(model=model).set(data["corr_fw"])

        # R² médio simples das três correlações
        r2_mean = float(np.mean([data["corr_lq"] ** 2, data["corr_cq"] ** 2, data["corr_fw"] ** 2]))
        all_r2.append(r2_mean)

        matrix = data["matrix"]
        labels = data["labels"]
        for i, label_x in enumerate(labels):
            for j, label_y in enumerate(labels):
                val = float(matrix[i][j]) if not math.isnan(matrix[i][j]) else 0.0
                model_correlation_matrix.labels(
                    model=model,
                    metric_x=label_x,
                    metric_y=label_y,
                ).set(val)

    if all_r2:
        nsga_correlation_r2_mean.set(float(np.mean(all_r2)))

# -----------------------------------------------------------------------------
# 💾 Persistência do histórico
# -----------------------------------------------------------------------------
def persist_correlations(corr_data: Dict[str, Dict[str, Any]]) -> None:
    """Salva correlações no banco de dados (correlation_history)."""
    if not corr_data:
        return

    rows = []
    for model, data in corr_data.items():
        r2_mean = float(np.mean([data["corr_lq"] ** 2, data["corr_cq"] ** 2, data["corr_fw"] ** 2]))
        rows.append(
            {
                "model": model,
                "corr_latency_quality": float(data["corr_lq"]),
                "corr_cost_quality": float(data["corr_cq"]),
                "corr_fitness_weights": float(data["corr_fw"]),
                "r2_mean": r2_mean,
                "generation": int(data["generation"]),
            }
        )

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return

    try:
        df_out.to_sql("correlation_history", db_engine, if_exists="append", index=False)
        logger.info(f"💾 Correlações salvas: {len(df_out)} registros")
    except Exception as e:
        logger.warning(f"⚠️ Erro ao salvar 'correlation_history': {e}")

# -----------------------------------------------------------------------------
# 🚀 Loop principal
# -----------------------------------------------------------------------------
def main() -> None:
    """Executa main."""
    logger.info(f"🚀 Servidor de correlações ativo (porta {PROM_PORT})")
    # Sobe o endpoint de métricas
    start_http_server(PROM_PORT)

    # Espera banco, garante tabela de histórico
    wait_for_db()
    ensure_history_table()

    # Loop
    while True:
        try:
            df = fetch_recent_metrics(window_sql="1 DAY")
            if df.empty:
                logger.warning(f"⚠️ Nenhum dado recente em 'model_metrics'. Aguardando {UPDATE_INTERVAL}s.")
                time.sleep(UPDATE_INTERVAL)
                continue

            corr = compute_correlations(df)
            if not corr:
                logger.warning(f"⚠️ Sem grupos/modelos suficientes para correlação. Aguardando {UPDATE_INTERVAL}s.")
                time.sleep(UPDATE_INTERVAL)
                continue

            publish_metrics(corr)
            persist_correlations(corr)

            logger.info(f"✅ Correlações publicadas e salvas — {len(corr)} modelos ({datetime.now()})")
        except Exception as e:
            logger.error(f"❌ Erro no loop de correlações: {e}")

        time.sleep(UPDATE_INTERVAL)

# -----------------------------------------------------------------------------
# ▶️ Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
