"""
correlation_metrics.py — Cálculo, exposição e armazenamento histórico das correlações multiobjetivo (NSGA-II)
Autor: Jefferson Duarte

Funcionalidades:
  ✅ Calcula correlações dinâmicas entre latência, custo, qualidade e fitness.
  ✅ Expõe métricas Prometheus (para dashboards).
  ✅ Armazena o histórico no banco MariaDB para análises futuras e auditorias.
"""

import os
import time
import numpy as np
import pandas as pd
import redis
from datetime import datetime
from prometheus_client import start_http_server, Gauge
from sqlalchemy import create_engine, text

# ==========================================================
# 🔧 Configurações de ambiente
# ==========================================================
DB_HOST = os.getenv("DB_HOST", "mariadb")
DB_USER = os.getenv("DB_USER", "router_user")
DB_PASS = os.getenv("DB_PASS", "router_pass")
DB_NAME = os.getenv("DB_NAME", "routerdb")

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASS = os.getenv("REDIS_PASSWORD", "SenhaForte")

PROM_PORT = int(os.getenv("PROM_PORT", 9105))
UPDATE_INTERVAL = int(os.getenv("CORR_INTERVAL", 60))

# ==========================================================
# 📊 Inicialização das métricas Prometheus
# ==========================================================
correlation_latency_quality = Gauge("correlation_latency_quality", "Correlação latência vs qualidade", ["model"])
correlation_cost_quality = Gauge("correlation_cost_quality", "Correlação custo vs qualidade", ["model"])
correlation_fitness_weights = Gauge("correlation_fitness_weights", "Correlação fitness vs pesos NSGA", ["model"])
nsga_correlation_r2_mean = Gauge("nsga_correlation_r2_mean", "Coeficiente R² médio global")
model_correlation_matrix = Gauge("model_correlation_matrix", "Matriz média de correlação (latência,custo,qualidade)", ["model", "metric_x", "metric_y"])

# ==========================================================
# 🔌 Conexões
# ==========================================================
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS, decode_responses=True)
db_engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

# ==========================================================
# 🧩 Funções utilitárias
# ==========================================================
def ensure_table_exists():
    """Cria tabela de histórico, se não existir."""
    ddl = """
    CREATE TABLE IF NOT EXISTS correlation_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        model VARCHAR(255),
        corr_latency_quality FLOAT,
        corr_cost_quality FLOAT,
        corr_fitness_weights FLOAT,
        r2_mean FLOAT,
        generation INT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """
    with db_engine.begin() as conn:
        conn.execute(text(ddl))

def fetch_recent_metrics():
    """Busca as métricas recentes do banco."""
    query = """
    SELECT model, latency_ms, cost_usd, quality_score, fitness, generation
    FROM model_metrics
    WHERE timestamp > NOW() - INTERVAL 1 DAY;
    """
    return pd.read_sql(query, db_engine)

def fetch_nsga_weights():
    """Busca pesos NSGA-II armazenados."""
    query = """
    SELECT model, weight_latency, weight_cost, weight_quality, generation
    FROM nsga_weights
    ORDER BY generation DESC
    LIMIT 300;
    """
    return pd.read_sql(query, db_engine)

def compute_correlations(df):
    """Calcula as correlações entre latência, custo, qualidade e fitness."""
    results = {}
    for model, group in df.groupby("model"):
        if len(group) < 3:
            continue

        lat = group["latency_ms"].astype(float)
        cost = group["cost_usd"].astype(float)
        qual = group["quality_score"].astype(float)
        fit = group["fitness"].astype(float)

        corr_lq = np.corrcoef(lat, qual)[0, 1]
        corr_cq = np.corrcoef(cost, qual)[0, 1]
        corr_fw = np.corrcoef(fit, cost + lat + qual)[0, 1]  # proxy de dependência composta

        matrix = np.corrcoef([lat, cost, qual])
        labels = ["latency", "cost", "quality"]

        results[model] = {
            "corr_lq": float(corr_lq),
            "corr_cq": float(corr_cq),
            "corr_fw": float(corr_fw),
            "matrix": (matrix, labels),
            "generation": int(group["generation"].max())
        }
    return results

def publish_metrics(corr_data):
    """Publica métricas Prometheus."""
    all_r2 = []
    for model, data in corr_data.items():
        correlation_latency_quality.labels(model=model).set(data["corr_lq"])
        correlation_cost_quality.labels(model=model).set(data["corr_cq"])
        correlation_fitness_weights.labels(model=model).set(data["corr_fw"])
        all_r2.append(np.mean([data["corr_lq"]**2, data["corr_cq"]**2, data["corr_fw"]**2]))

        matrix, labels = data["matrix"]
        for i, label_x in enumerate(labels):
            for j, label_y in enumerate(labels):
                model_correlation_matrix.labels(model=model, metric_x=label_x, metric_y=label_y).set(float(matrix[i][j]))

    if all_r2:
        nsga_correlation_r2_mean.set(np.mean(all_r2))

def persist_correlations(corr_data):
    """Salva correlações no banco de dados."""
    rows = []
    for model, data in corr_data.items():
        rows.append({
            "model": model,
            "corr_latency_quality": data["corr_lq"],
            "corr_cost_quality": data["corr_cq"],
            "corr_fitness_weights": data["corr_fw"],
            "r2_mean": np.mean([data["corr_lq"]**2, data["corr_cq"]**2, data["corr_fw"]**2]),
            "generation": data["generation"]
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_sql("correlation_history", db_engine, if_exists="append", index=False)
        print(f"💾 Correlações salvas: {len(df)} registros")

# ==========================================================
# 🚀 Loop principal
# ==========================================================
def main():
    print(f"🚀 Servidor de correlações ativo (porta {PROM_PORT})")
    start_http_server(PROM_PORT)
    ensure_table_exists()

    while True:
        try:
            df = fetch_recent_metrics()
            if df.empty:
                print("⚠️ Nenhum dado recente encontrado.")
                time.sleep(UPDATE_INTERVAL)
                continue

            corr = compute_correlations(df)
            publish_metrics(corr)
            persist_correlations(corr)

            print(f"✅ Correlações publicadas e salvas — {len(corr)} modelos ({datetime.now()})")
        except Exception as e:
            print(f"❌ Erro ao calcular correlações: {e}")
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    main()
