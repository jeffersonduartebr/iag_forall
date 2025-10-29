# nsga_weights_updater.py
import os
import json
import random
import logging
import numpy as np
from datetime import datetime
from deap import base, creator, tools, algorithms

# ======================================================
# CONFIGURAÇÃO GLOBAL
# ======================================================
DATA_DIR = "/app/app/data"
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nsga-updater")

# ======================================================
# FUNÇÕES AUXILIARES
# ======================================================

def _normalize(x: float, max_val: float) -> float:
    return max(0.0, min(1.0, x / max_val))

def _load_history():
    if not os.path.exists(HISTORY_FILE):
        logger.warning(f"[nsga] Nenhum histórico encontrado em {HISTORY_FILE}.")
        return []
    with open(HISTORY_FILE, "r") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Formato inesperado em history.json")
            return data
        except Exception as e:
            logger.error(f"[nsga] Falha ao ler histórico: {e}")
            return []

def _aggregate_performance(records):
    """
    Converte o histórico do Bandit em métricas médias agregadas.
    """
    by_model = {}
    for r in records:
        model = r.get("model")
        reward = float(r.get("reward", 0.0))
        ema = float(r.get("ema", 0.0))
        if not model:
            continue
        if model not in by_model:
            by_model[model] = {"rewards": []}
        by_model[model]["rewards"].append(reward)

    metrics = {}
    for model, d in by_model.items():
        rewards = d["rewards"]
        metrics[model] = {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards))
        }
    return metrics

# ======================================================
# NSGA-II CONFIGURAÇÃO
# ======================================================

def nsga_fitness(individual, metrics):
    """
    Função objetivo: gera fitness simulada com base nos dados históricos.
    """
    wq, wc, wl = individual
    # peso de qualidade maior → recompensa esperada maior
    # mas penaliza custo e latência indiretamente (simulado)
    avg_reward = np.mean([m["reward_mean"] for m in metrics.values()])
    f_quality = avg_reward * wq
    f_cost = wc  # queremos minimizar
    f_latency = wl  # queremos minimizar
    return f_quality, -f_cost, -f_latency

def run_nsga_ii(history_metrics, ngen=20, pop_size=50):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", nsga_fitness, metrics=history_metrics)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, 
                              cxpb=0.5, mutpb=0.3, ngen=ngen, verbose=False)

    fronts = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    best = max(fronts, key=lambda ind: ind.fitness.values[0])  # melhor qualidade
    return best, fronts

# ======================================================
# EXECUÇÃO PRINCIPAL
# ======================================================

def main():
    logger.info("[nsga] Iniciando atualização de pesos...")
    records = _load_history()
    if not records:
        logger.warning("[nsga] Nenhum dado para otimização. Encerrando.")
        return

    metrics = _aggregate_performance(records)
    best, pareto = run_nsga_ii(metrics, ngen=20, pop_size=30)
    best_weights = [round(float(x), 3) for x in best]
    logger.info(f"[nsga] Novo vetor ótimo: {best_weights}")

    pareto_points = [
        {"theta": [round(float(x), 3) for x in ind], "fitness": list(ind.fitness.values)}
        for ind in pareto
    ]

    new_data = {
        "generation": datetime.utcnow().isoformat(),
        "current_best": best_weights,
        "pareto_front": pareto_points
    }

    with open(WEIGHTS_FILE, "w") as f:
        json.dump(new_data, f, indent=2)

    logger.info(f"[nsga] Atualização concluída. Arquivo salvo em {WEIGHTS_FILE}")

if __name__ == "__main__":
    main()
