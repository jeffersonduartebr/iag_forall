# -*- coding: utf-8 -*-
"""Policy helpers for meta-bandit decisions."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np


def dynamic_epsilon(ctx_stats: Dict[str, Dict[str, float]], default_epsilon: float) -> float:
    """Resumo do comportamento desta função.

    Args:
        ctx_stats: Parâmetro de entrada.
        default_epsilon: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    eps = default_epsilon
    if not ctx_stats:
        return min(1.0, eps + 0.15)

    counts = [s.get("count", 0) for s in ctx_stats.values()]
    if counts and min(counts) < 3:
        eps += 0.10

    vars_ = [s.get("var", 0.0) for s in ctx_stats.values()]
    if vars_ and float(np.mean(vars_)) > 0.05:
        eps += 0.08

    return max(0.0, min(1.0, eps))


def choose_epsilon_greedy(
    models: List[str],
    ctx_stats: Dict[str, Dict[str, float]],
    default_epsilon: float,
) -> str:
    """Resumo do comportamento desta função.

    Args:
        models: Parâmetro de entrada.
        ctx_stats: Parâmetro de entrada.
        default_epsilon: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    eps = dynamic_epsilon(ctx_stats, default_epsilon)
    if random.random() < eps:
        scored = []
        for m in models:
            s = ctx_stats.get(m, {})
            cnt = s.get("count", 0)
            var = s.get("var", 0.0)
            expl = (1.0 / (1.0 + cnt)) + 0.5 * var
            scored.append((m, expl))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    scored = []
    for m in models:
        s = ctx_stats.get(m, {})
        mean = s.get("mean", 0.0)
        scored.append((m, mean))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def choose_ucb1(models: List[str], ctx_stats: Dict[str, Dict[str, float]]) -> str:
    """Resumo do comportamento desta função.

    Args:
        models: Parâmetro de entrada.
        ctx_stats: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    total = sum(s.get("count", 0) for s in ctx_stats.values())
    if total <= 0:
        total = 1

    c = 1.4
    scores = []
    for m in models:
        s = ctx_stats.get(m, {})
        mean = s.get("mean", 0.0)
        cnt = s.get("count", 0)
        if cnt <= 0:
            bonus = float("inf")
        else:
            bonus = c * math.sqrt(math.log(total + 1.0) / cnt)
        scores.append((m, mean + bonus))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]


def choose_thompson(models: List[str], ctx_stats: Dict[str, Dict[str, float]]) -> str:
    """Resumo do comportamento desta função.

    Args:
        models: Parâmetro de entrada.
        ctx_stats: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    cand = []
    for m in models:
        s = ctx_stats.get(m, {})
        alpha = float(s.get("alpha", 1.0))
        beta = float(s.get("beta", 1.0))
        if alpha <= 0 or beta <= 0 or not math.isfinite(alpha) or not math.isfinite(beta):
            alpha, beta = 1.0, 1.0
        sample = np.random.beta(alpha, beta)
        cand.append((m, float(sample)))

    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[0][0]


def meta_combine_choices(
    models: List[str],
    ctx_stats: Dict[str, Dict[str, float]],
    default_epsilon: float,
    preferred_strategy: str,
) -> Tuple[str, Dict[str, str]]:
    """Resumo do comportamento desta função.

    Args:
        models: Parâmetro de entrada.
        ctx_stats: Parâmetro de entrada.
        default_epsilon: Parâmetro de entrada.
        preferred_strategy: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    if not models:
        raise RuntimeError("Nenhum modelo recebido em meta_combine_choices.")

    eps_choice = choose_epsilon_greedy(models, ctx_stats, default_epsilon)
    ucb_choice = choose_ucb1(models, ctx_stats)
    ts_choice = choose_thompson(models, ctx_stats)

    votes: Dict[str, int] = {}
    for m in (eps_choice, ucb_choice, ts_choice):
        votes[m] = votes.get(m, 0) + 1

    best_model = None
    best_votes = -1
    for m, v in votes.items():
        if v > best_votes:
            best_model = m
            best_votes = v

    if best_votes >= 2:
        chosen = best_model
    else:
        if preferred_strategy == "epsilon_greedy":
            chosen = eps_choice
        elif preferred_strategy == "thompson":
            chosen = ts_choice
        else:
            chosen = ucb_choice

    return chosen, {
        "epsilon_greedy": eps_choice,
        "ucb1": ucb_choice,
        "thompson": ts_choice,
    }
