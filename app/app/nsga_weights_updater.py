# app/nsga_weights_updater.py
"""
Serviço NSGA-II autônomo para otimização dos pesos de recompensa do router.
Agora inclui:
✅ Integração com MariaDB
✅ Servidor Prometheus embutido (porta 8000)
✅ Resiliência e logging estruturado
✅ Atualizações periódicas automáticas
"""

import os
import time
import json
import random
import logging
import numpy as np
from datetime import datetime
from prometheus_client import (
    Counter, Gauge, Histogram, start_http_server
)
from app.db_manager import load_history, insert_weights, get_conn

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
# PARÂMETROS DO ALGORITMO NSGA-II
# ============================================================
POP_SIZE = 100
MAX_GEN = 25
MUT_RATE = 0.15
CROSS_RATE = 0.6
EPS = 1e-6
DEFAULT_THETA = [0.55, 0.25, 0.20]
RETRY_DELAY = 15
MAX_DB_RETRIES = 5
SLEEP_BETWEEN_RUNS = 7200  # 2h

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def _fitness(rewards: list[float]) -> float:
    """Calcula fitness médio ponderado."""
    rewards = np.array(rewards, dtype=np.float32)
    return float(np.mean(rewards))

def _mutate(theta: list[float]) -> list[float]:
    """Aplica mutação leve em um gene e renormaliza."""
    theta = np.array(theta)
    i = random.randint(0, len(theta) - 1)
    theta[i] = max(0, min(1, theta[i] + np.random.uniform(-0.1, 0.1)))
    theta /= np.sum(theta)
    return theta.tolist()

def _crossover(p1: list[float], p2: list[float]) -> list[float]:
    """Cruzamento linear simples entre dois indivíduos."""
    α = np.random.rand()
    child = α * np.array(p1) + (1 - α) * np.array(p2)
    child /= np.sum(child)
    return child.tolist()

def _ensure_db_connection():
    """Verifica e tenta reconectar ao banco de dados até MAX_DB_RETRIES vezes."""
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
def run_nsga():
    start_time = time.time()
    logger.info("[nsga] Iniciando atualização de pesos...")

    # 1️⃣ Verifica conexão com o banco
    if not _ensure_db_connection():
        logger.error("[nsga] Abortando ciclo por falta de conexão com o banco.")
        return

    # 2️⃣ Carrega histórico do banco
    try:
        history = load_history(limit=1000)
    except Exception as e:
        NSGA_DB_ERRORS.inc()
        logger.error(f"[nsga] Erro ao carregar histórico do banco: {e}")
        return

    if not history:
        logger.warning("[nsga] Nenhum dado disponível para otimização. Encerrando ciclo.")
        return

    # 3️⃣ Extrai recompensas válidas
    rewards = [h["reward"] for h in history if h.get("reward") is not None]
    if len(rewards) < 10:
        logger.warning(f"[nsga] Histórico insuficiente ({len(rewards)} registros). Aguardando mais dados.")
        return

    # 4️⃣ Inicializa população aleatória
    population = [np.random.dirichlet(np.ones(3)).tolist() for _ in range(POP_SIZE)]
    best_theta = DEFAULT_THETA
    best_fit = -float("inf")

    # 5️⃣ Loop de gerações
    for gen in range(MAX_GEN):
        fitness_scores = []
        for θ in population:
            sim_rewards = [r * (θ[0] - θ[1]*0.2 - θ[2]*0.1) for r in rewards]
            fit = _fitness(sim_rewards)
            fitness_scores.append(fit)

        mean_fit = float(np.mean(fitness_scores))
        best_idx = int(np.argmax(fitness_scores))

        if fitness_scores[best_idx] > best_fit + EPS:
            best_fit = fitness_scores[best_idx]
            best_theta = population[best_idx]

        logger.info(f"[nsga] Geração {gen+1}/{MAX_GEN} | Fit médio={mean_fit:.3f} | Melhor={best_fit:.3f}")

        NSGA_GENERATIONS.inc()
        NSGA_LAST_FITNESS.set(mean_fit)

        # Evolução
        new_pop = []
        for _ in range(POP_SIZE // 2):
            p1, p2 = random.choices(population, k=2)
            c1, c2 = _crossover(p1, p2), _crossover(p2, p1)
            if random.random() < MUT_RATE: c1 = _mutate(c1)
            if random.random() < MUT_RATE: c2 = _mutate(c2)
            new_pop.extend([c1, c2])
        population = new_pop

    # 6️⃣ Persiste pesos no banco
    try:
        insert_weights(best_theta, best_fit, MAX_GEN)
        logger.info(f"[nsga] Pesos persistidos: θ={best_theta} | Fitness={best_fit:.4f}")
    except Exception as e:
        NSGA_DB_ERRORS.inc()
        logger.error(f"[nsga] Falha ao gravar pesos otimizados: {e}")
        return

    # 7️⃣ Métricas finais
    elapsed = time.time() - start_time
    NSGA_LAST_UPDATE.set(elapsed)
    NSGA_EXECUTION_TIME.observe(elapsed)
    logger.info(f"[nsga] Atualização concluída em {elapsed:.2f}s ✅")

# ============================================================
# LOOP PRINCIPAL + SERVIDOR PROMETHEUS
# ============================================================
if __name__ == "__main__":
    logger.info("[nsga] Iniciando servidor Prometheus na porta 8000...")
    start_http_server(8000)
    logger.info("[nsga] Servidor Prometheus ativo. Métricas disponíveis em /metrics")

    while True:
        try:
            run_nsga()
        except Exception as e:
            logger.exception(f"[nsga] Erro inesperado no ciclo principal: {e}")
        logger.info(f"[nsga] Aguardando {SLEEP_BETWEEN_RUNS/3600:.1f}h para próxima execução...")
        time.sleep(SLEEP_BETWEEN_RUNS)
