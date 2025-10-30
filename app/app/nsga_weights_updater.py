# app/nsga_weights_updater.py
import os
import time
import re
import json
import random
import logging
import numpy as np
from datetime import datetime
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from .db_manager import load_history, insert_weights, get_conn

# ============================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nsga-updater")

# ============================================================
# MÉTRICAS PROMETHEUS
# ============================================================
NSGA_GENERATIONS = Counter("nsga_generations_total", "Número total de gerações NSGA executadas.")
NSGA_LAST_FITNESS = Gauge("nsga_last_fitness_mean", "Fitness médio da última geração NSGA.")
NSGA_LAST_UPDATE = Gauge("nsga_weights_last_update_seconds", "Tempo desde a última atualização de pesos NSGA (s).")
NSGA_EXECUTION_TIME = Histogram("nsga_execution_time_seconds", "Tempo de execução de cada ciclo de otimização NSGA.")
NSGA_DB_ERRORS = Counter("nsga_db_errors_total", "Número de erros de conexão ou escrita no banco de dados.")

# ============================================================
# PARÂMETROS DO NSGA-II
# ============================================================
POP_SIZE = 200         # população ampliada
MAX_GEN = 50           # número de gerações
MUT_RATE = 0.15
CROSS_RATE = 0.6
EPS = 1e-6
RETRY_DELAY = 15
MAX_DB_RETRIES = 5
SLEEP_BETWEEN_RUNS = 7200  # 2 horas entre ciclos

# ============================================================
# CONFIGURAÇÕES POR FAMÍLIA DE MODELO
# ============================================================
MODEL_FAMILIES = {
    "phi": {"default_weights": [0.65, 0.20, 0.15], "cost_target": 0.002, "latency_target": 1.3, "token_key": "max_tokens"},
    "deepseek": {"default_weights": [0.55, 0.25, 0.20], "cost_target": 0.005, "latency_target": 1.4, "token_key": "max_tokens"},
    "llama": {"default_weights": [0.60, 0.25, 0.15], "cost_target": 0.008, "latency_target": 1.5, "token_key": "max_tokens"},
    "mistral": {"default_weights": [0.58, 0.22, 0.20], "cost_target": 0.006, "latency_target": 1.1, "token_key": "max_tokens"},
    "qwen": {"default_weights": [0.50, 0.30, 0.20], "cost_target": 0.009, "latency_target": 1.2, "token_key": "max_tokens"},
    "gemma": {"default_weights": [0.57, 0.23, 0.20], "cost_target": 0.010, "latency_target": 1.0, "token_key": "max_tokens"},
    "gpt": {"default_weights": [0.45, 0.35, 0.20], "cost_target": 0.12, "latency_target": 0.9, "token_key": "max_completion_tokens"},
    "gemini": {"default_weights": [0.50, 0.30, 0.20], "cost_target": 0.15, "latency_target": 1.0, "token_key": "max_tokens"},
    "default": {"default_weights": [0.55, 0.25, 0.20], "cost_target": 0.01, "latency_target": 1.5, "token_key": "max_tokens"},
}

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def detect_model_family(model_name: str) -> str:
    """
    Identifica dinamicamente a família real do modelo (phi, llama, deepseek, gpt, gemini etc.)
    mesmo quando servido via Ollama ou outro provedor.
    """
    if not model_name:
        return "default"
    model_name = model_name.lower()
    for fam in MODEL_FAMILIES.keys():
        if fam in model_name:
            return fam
    return "default"

def _fitness(rewards: list[float]) -> float:
    return float(np.mean(np.array(rewards, dtype=np.float32)))

def _mutate(theta: list[float]) -> list[float]:
    theta = np.array(theta, dtype=np.float32)
    i = random.randint(0, len(theta) - 1)
    theta[i] = max(0, min(1, float(theta[i] + np.random.uniform(-0.1, 0.1))))
    s = float(np.sum(theta))
    theta = theta / s if s > 0 else np.array([1/3, 1/3, 1/3], dtype=np.float32)
    return theta.tolist()

def _crossover(p1: list[float], p2: list[float]) -> list[float]:
    a = float(np.random.rand())
    child = a * np.array(p1, dtype=np.float32) + (1 - a) * np.array(p2, dtype=np.float32)
    s = float(np.sum(child))
    child = child / s if s > 0 else np.array([1/3, 1/3, 1/3], dtype=np.float32)
    return child.tolist()

def _ensure_db_connection():
    for attempt in range(1, MAX_DB_RETRIES + 1):
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
            logger.info("[nsga] Conexão com o banco de dados verificada com sucesso.")
            return True
        except Exception as e:
            logger.warning(f"[nsga] Falha ao conectar ao banco (tentativa {attempt}/{MAX_DB_RETRIES}): {e}")
            NSGA_DB_ERRORS.inc()
            time.sleep(RETRY_DELAY)
    logger.error("[nsga] Banco de dados indisponível após múltiplas tentativas.")
    return False

# ============================================================
# NÚCLEO DO NSGA-II
# ============================================================
def run_nsga(model_name: str = "ollama/deepseek-r1:1.5b"):
    time.sleep(10)  # breve pausa para evitar sobrecarga
    start_time = time.time()
    model_family = detect_model_family(model_name)
    fam_cfg = MODEL_FAMILIES.get(model_family, MODEL_FAMILIES["default"])

    logger.info(
        f"[nsga] Iniciando ciclo para '{model_name}' | família={model_family} "
        f"(token_key={fam_cfg['token_key']}, cost*={fam_cfg['cost_target']}, lat*={fam_cfg['latency_target']})"
    )

    # 1) Verifica DB
    if not _ensure_db_connection():
        logger.error("[nsga] Abortando ciclo por falta de conexão com o banco.")
        return

    # 2) Coleta de dados (métricas dinâmicas + fallback banco) — ROBUSTO
    rewards: list[float] = []
    try:
        live_metrics = {}
        try:
            from .metrics_collector import get_snapshot
            live_metrics = get_snapshot() or {}
        except Exception as e:
            logger.warning(f"[nsga] metrics_collector indisponível: {e}")

        if live_metrics:
            logger.info(f"[nsga] Coletando métricas dinâmicas de {len(live_metrics)} modelos...")
            for _, m in live_metrics.items():
                # qualidade alta e custos/latência baixos => reward maior
                q = float(m.get("quality", 0.0))
                l = float(m.get("latency", 0.0))
                c = float(m.get("cost", 0.0))
                reward = q / (1.0 + l + (c * 10.0))
                rewards.append(reward)
        else:
            logger.warning("[nsga] Nenhuma métrica dinâmica disponível. Tentando histórico no banco...")
            history = load_history(limit=1000) or []
            rewards = [float(h["reward"]) for h in history if h and h.get("reward") is not None]

    except Exception as e:
        NSGA_DB_ERRORS.inc()
        logger.error(f"[nsga] Erro ao coletar métricas ou histórico: {e}")
        rewards = []

    # 3) Validação de dados
    if not rewards or len(rewards) < 10:
        logger.warning(f"[nsga] Dados insuficientes para otimização ({len(rewards)} registros do modelo {model_name}). Encerrando ciclo.")
        return

    # 4) Inicialização da população
    population = [np.random.dirichlet(np.ones(3)).tolist() for _ in range(POP_SIZE)]
    best_theta = fam_cfg["default_weights"]
    best_fit = -float("inf")

    # 5) Loop evolutivo
    for gen in range(MAX_GEN):
        fitness_scores = []
        for th in population:
            # proxy simples de utilidade ponderada
            sim_rewards = [r * (th[0] - th[1]*0.2 - th[2]*0.1) for r in rewards]
            fitness_scores.append(_fitness(sim_rewards))

        mean_fit = float(np.mean(fitness_scores))
        best_idx = int(np.argmax(fitness_scores))
        if fitness_scores[best_idx] > best_fit + EPS:
            best_fit = fitness_scores[best_idx]
            best_theta = population[best_idx]

        logger.info(f"[nsga] Geração {gen+1}/{MAX_GEN} | Fit médio={mean_fit:.3f} | Melhor={best_fit:.3f}")
        NSGA_GENERATIONS.inc()
        NSGA_LAST_FITNESS.set(mean_fit)

        # reprodução
        new_pop = []
        for _ in range(POP_SIZE // 2):
            p1, p2 = random.choices(population, k=2)
            c1, c2 = _crossover(p1, p2), _crossover(p2, p1)
            if random.random() < MUT_RATE:
                c1 = _mutate(c1)
            if random.random() < MUT_RATE:
                c2 = _mutate(c2)
            new_pop.extend([c1, c2])
        population = new_pop

    # 6) Persistência
    try:
        insert_weights(
            best_theta, best_fit, MAX_GEN,
            model_name=model_name,
            model_family=model_family,
            token_key=fam_cfg["token_key"]
        )
        logger.info(f"[nsga] Pesos persistidos para {model_name}({model_family}) | θ={best_theta} | Fitness={best_fit:.4f}")
    except Exception as e:
        NSGA_DB_ERRORS.inc()
        logger.error(f"[nsga] Falha ao gravar pesos otimizados: {e}")

    # 7) Métricas de execução
    elapsed = time.time() - start_time
    NSGA_LAST_UPDATE.set(elapsed)
    NSGA_EXECUTION_TIME.observe(elapsed)
    logger.info(f"[nsga] Atualização concluída em {elapsed:.2f}s ✅")

# ============================================================
# LOOP PRINCIPAL
# ============================================================
if __name__ == "__main__":
    logger.info("[nsga] Servidor Prometheus na porta 9999...")
    start_http_server(9999)

    while True:
        try:
            # Execute para múltiplos “model identifiers” (provedor ≠ família)
            for model in [
                "ollama/deepseek-r1:8b",
                "ollama/phi4",
                "ollama/llama3:8b",
                "openai/gpt-5-nano",
                "gemini/gemini-2.0-flash",
            ]:
                run_nsga(model)
                time.sleep(5)
        except Exception as e:
            logger.exception(f"[nsga] Erro inesperado no ciclo principal: {e}")
        logger.info(f"[nsga] Aguardando {SLEEP_BETWEEN_RUNS/3600:.1f}h para próxima execução...")
        time.sleep(SLEEP_BETWEEN_RUNS)
