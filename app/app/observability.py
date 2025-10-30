"""
observability.py
----------------------------------------------------
Gerencia métricas Prometheus (modo multiprocess ou single-process)
e logging estruturado via Structlog.
Evita FileNotFoundError e falhas na inicialização.
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
# ⚙️ Limpeza e preparação segura do diretório multiprocess
# ============================================================

def _prepare_prometheus_dir():
    """
    Garante que o diretório PROMETHEUS_MULTIPROC_DIR exista e esteja limpo.
    Evita falhas 'FileNotFoundError' durante o bootstrap.
    """
    prom_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "/tmp/prom")

    try:
        if not os.path.exists(prom_dir):
            os.makedirs(prom_dir, exist_ok=True)
            print(f"[observability] Diretório criado: {prom_dir}")
        else:
            for f in os.listdir(prom_dir):
                fpath = os.path.join(prom_dir, f)
                try:
                    os.remove(fpath)
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"[observability] Aviso: falha ao limpar {fpath}: {e}")
    except Exception as e:
        print(f"[observability] Erro ao preparar diretório multiprocess: {e}")

_prepare_prometheus_dir()

# ============================================================
# 🧠 Inicialização segura do Prometheus Collector
# ============================================================

try:
    registry = CollectorRegistry()
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        multiprocess.MultiProcessCollector(registry)
        print("[observability] Prometheus multiprocess configurado com sucesso.")
    else:
        print("[observability] Prometheus operando em modo single-process.")
except Exception as e:
    print(f"[observability] Falha ao inicializar Prometheus multiprocess: {e}")
    registry = CollectorRegistry()

# ============================================================
# 📈 Definição das métricas
# ============================================================

# 🔹 API
API_REQUESTS = Counter(
    "api_requests_total",
    "Total de requisições recebidas pela API",
    registry=registry
)
API_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latência das requisições da API (s)",
    registry=registry
)

# 🔹 Roteamento e candidatos
ROUTER_CHOSEN = Counter(
    "router_chosen_model_total",
    "Modelo escolhido pelo roteador",
    ["model"],
    registry=registry
)
FALLBACK_USED = Counter(
    "router_fallback_used_total",
    "Fallback usado entre modelos",
    ["first_model", "second_model"],
    registry=registry
)
CANDIDATE_COST = Histogram(
    "candidate_estimated_cost_usd",
    "Custo estimado por resposta (USD)",
    registry=registry
)
CANDIDATE_LAT = Histogram(
    "candidate_latency_seconds",
    "Latência por resposta (s)",
    registry=registry
)
JUDGE_SCORE = Histogram(
    "judge_score",
    "Valores de pontuação dos juízes",
    ["judge_id"],
    registry=registry
)
ROUTER_HISTORY_ENTRIES = Gauge(
    "router_history_entries_total",
    "Número de modelos com histórico EMA",
    registry=registry
)

# 🔹 Bandit (aprendizado adaptativo)
BANDIT_SELECT = Counter(
    "bandit_select_total",
    "Seleções do bandit por modelo",
    ["model"],
    registry=registry
)
BANDIT_UPDATE = Counter(
    "bandit_update_total",
    "Atualizações de bandit por modelo",
    ["model"],
    registry=registry
)
BANDIT_REWARD = Histogram(
    "bandit_reward",
    "Valores observados de recompensa do bandit",
    registry=registry
)

# 🔹 Custos e qualidade
ROUTER_MODEL_COST = Counter(
    "router_model_cost_usd_total",
    "Custo acumulado estimado por modelo (USD)",
    ["model"],
    registry=registry
)
ROUTER_COST_SAVINGS = Counter(
    "router_cost_savings_usd_total",
    "Custo total economizado (USD)",
    registry=registry
)
ROUTER_COST_PER_QUERY = Gauge(
    "router_cost_per_query_usd",
    "Custo médio por consulta (USD)",
    registry=registry
)
ROUTER_QUALITY_AVG = Gauge(
    "router_model_quality_avg",
    "Qualidade média ponderada por modelo (0–10)",
    ["model"],
    registry=registry
)
ROUTER_LOCAL_USAGE_RATIO = Gauge(
    "router_local_model_usage_ratio",
    "Proporção de requisições atendidas por modelos locais (Ollama)",
    registry=registry
)

# ============================================================
# 🪵 Logging estruturado com Structlog
# ============================================================

logger = structlog.get_logger("observability")

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

