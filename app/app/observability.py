"""
observability.py
----------------------------------------------------
Gerencia métricas Prometheus (modo multiprocess) e logging estruturado.
Compatível com Uvicorn + Gunicorn + Prometheus-client multiprocess.
Evita FileNotFoundError ao limpar diretórios temporários.
"""

import os
import logging
import structlog
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
    multiprocess
)

# ============================================================
# ⚙️ Inicialização do Prometheus multiprocess-safe
# ============================================================
logger = structlog.get_logger("observability")

PROM_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "/tmp/prom")

# Garante que o diretório exista
os.makedirs(PROM_DIR, exist_ok=True)

# Limpa arquivos antigos, mas ignora ausentes
try:
    for f in os.listdir(PROM_DIR):
        path = os.path.join(PROM_DIR, f)
        try:
            os.remove(path)
        except FileNotFoundError:
            continue  # arquivo já foi removido
        except Exception as e:
            logger.warning(f"[observability] Falha ao remover {path}: {e}")
except Exception as e:
    logger.warning(f"[observability] Falha ao limpar PROMETHEUS_MULTIPROC_DIR: {e}")

# ============================================================
# 📈 Métricas principais
# ============================================================

# 🔹 Requisições de API
API_REQUESTS = Counter("api_requests_total", "Total de requisições recebidas pela API")
API_LATENCY = Histogram("api_request_latency_seconds", "Latência das requisições da API (s)")

# 🔹 Decisões do roteador
ROUTER_CHOSEN = Counter("router_chosen_model_total", "Modelo escolhido pelo roteador", ["model"])
FALLBACK_USED = Counter("router_fallback_used_total", "Fallback usado", ["first_model", "second_model"])
CANDIDATE_COST = Histogram("candidate_estimated_cost_usd", "Custo estimado por resposta (USD)")
CANDIDATE_LAT = Histogram("candidate_latency_seconds", "Latência por resposta (s)")
JUDGE_SCORE = Histogram("judge_score", "Valores de pontuação dos juízes", ["judge_id"])

# 🔹 Histórico e Bandits
ROUTER_HISTORY_ENTRIES = Gauge("router_history_entries_total", "Número de modelos com histórico EMA")
BANDIT_SELECT = Counter("bandit_select_total", "Seleções do bandit por modelo", ["model"])
BANDIT_UPDATE = Counter("bandit_update_total", "Atualizações de bandit por modelo", ["model"])
BANDIT_REWARD = Histogram("bandit_reward", "Valores observados de recompensa do bandit")

# 💰 Custos e economia
ROUTER_MODEL_COST = Counter(
    "router_model_cost_usd_total",
    "Custo acumulado estimado por modelo (USD)",
    ["model"]
)

ROUTER_COST_SAVINGS = Counter(
    "router_cost_savings_usd_total",
    "Custo total economizado (USD) pelo uso de modelos locais."
)

ROUTER_COST_PER_QUERY = Gauge(
    "router_cost_per_query_usd",
    "Custo médio por consulta (USD)."
)

# ⚖️ Qualidade e desempenho
ROUTER_QUALITY_AVG = Gauge(
    "router_model_quality_avg",
    "Qualidade média ponderada por modelo (0–10)",
    ["model"]
)

# ⚙️ Distribuição local vs remoto
ROUTER_LOCAL_USAGE_RATIO = Gauge(
    "router_local_model_usage_ratio",
    "Proporção de requisições atendidas por modelos locais (Ollama)."
)

# ============================================================
# 🪵 Logging estruturado
# ============================================================
def setup_logging():
    """Configura o Structlog para JSON com timestamp e nível de log."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ]
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger.info("[observability] Logging configurado com sucesso.")
