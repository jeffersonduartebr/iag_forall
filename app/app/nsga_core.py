# -*- coding: utf-8 -*-
# Objective: NSGA-II core (DEAP) and convergence metrics, free of I/O.
"""Pure pieces of the NSGA-II updater.

``run_nsga_optimization`` searches the per-model portfolio weights over
(latency, cost, quality, alignment) with DEAP's NSGA-II and returns the best
individual's weights plus the system metrics at that operating point;
``compute_convergence_metrics`` scores the efficiency history. Extracted from
``app.nsga_weights_updater`` (re-exported there).
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
from deap import algorithms, base, creator, tools


def compute_convergence_metrics(history: List[float]) -> Dict[str, float]:
    """
    Compute convergence metrics from efficiency history.

    Returns:
        Dict with trend, variance, and health score
    """
    if len(history) < 3:
        return {"trend": 0.0, "variance": 0.0, "health": 1.0}

    # Compute variance (lower is better for convergence)
    variance = float(np.var(history))

    # Compute trend (positive = improving, negative = degrading)
    # Using simple linear regression slope
    x = np.arange(len(history))
    slope = float(np.polyfit(x, history[::-1], 1)[0])  # Reverse because newest is first

    # Compute health score
    # Health degrades if:
    # 1. High variance (unstable)
    # 2. Negative trend (degrading)
    # 3. Efficiency too low

    avg_efficiency = float(np.mean(history))

    # Health scoring logic
    if variance > 1.0 or slope < -0.5:
        health = -1.0  # Stuck or diverging
    elif variance > 0.5 or slope < -0.1 or avg_efficiency < 1.0:
        health = 0.0  # Degraded
    else:
        health = 1.0  # Healthy

    return {
        "trend": slope,
        "variance": variance,
        "health": health,
        "avg_efficiency": avg_efficiency,
    }


def run_nsga_optimization(
    modality: str, models: List[str], metrics: Dict[str, Dict[str, float]], n_pop=40, n_gen=20
) -> Tuple[Dict[str, float], float, Tuple[float, float, float]]:
    """
    Roda o NSGA-II.
    Retorna: (pesos_modelos, pontuação_eficiência, (lat_media, cost_medio, qual_media))
    """
    n = len(models)
    if n == 0:
        return {}, 0.0, (0, 0, 0)
    if n == 1:
        return (
            {models[0]: 1.0},
            1.0,
            (metrics[models[0]]["latency"], metrics[models[0]]["cost"], metrics[models[0]]["quality"]),
        )

    # Limpa classes anteriores
    if "FitnessMulti" in creator.__dict__:
        del creator.FitnessMulti
    if "Individual" in creator.__dict__:
        del creator.Individual

    # Objetivos fixos para o AG: Min Latency, Min Cost, Max Quality, Max Alignment
    # Usamos pesos fixos AQUI para encontrar o Pareto Front ideal matemático.
    # O ajuste dinâmico será feito nos pesos do ROUTER, baseado no resultado daqui.
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -50.0, 2.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        """Execute the evaluate routine.

        This helper encapsulates one focused step used by the surrounding workflow."""
        s = sum(individual) or 1.0
        w = [x / s for x in individual]

        lat = sum(w[i] * metrics[models[i]]["latency"] for i in range(n))
        cst = sum(w[i] * metrics[models[i]]["cost"] for i in range(n))
        qlt = sum(w[i] * metrics[models[i]]["quality"] for i in range(n))
        aln = sum(w[i] * metrics[models[i]]["alignment"] for i in range(n))
        return lat, cst, qlt, aln

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0 / n)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=n_pop)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=n_pop, lambda_=n_pop, cxpb=0.9, mutpb=0.1, ngen=n_gen, verbose=False)

    best_ind = tools.selBest(pop, 1)[0]
    s = sum(best_ind) or 1.0
    norm_weights = [x / s for x in best_ind]

    weights_map = {models[i]: norm_weights[i] for i in range(n)}

    # Métricas do sistema ideal encontrado
    sys_lat, sys_cst, sys_qlt, _ = evaluate(best_ind)
    efficiency_score = sys_qlt / max(0.01, sys_lat)

    return weights_map, efficiency_score, (sys_lat, sys_cst, sys_qlt)
