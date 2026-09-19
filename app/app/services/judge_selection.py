# -*- coding: utf-8 -*-
# Objective: Adaptive selection of the judge pair.
"""Pick two judge models by recent fitness and quality/cost, with epsilon exploration.

Extracted from ``app.judges`` (re-exported there).
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..settings_dynamic import settings


def _safe_setting_float(key: str, default: float) -> float:
    """Read one float setting defensively for judge configuration."""
    try:
        return float(settings.get(key, default))
    except Exception:
        return float(default)


MIN_FITNESS = _safe_setting_float("JUDGES_MIN_FITNESS", 0.30)
W_FIT = _safe_setting_float("JUDGES_WEIGHT_FITNESS", 0.6)
W_QC = _safe_setting_float("JUDGES_WEIGHT_QC", 0.4)
EPSILON_RANDOM = _safe_setting_float("JUDGES_EPSILON", 0.10)


@dataclass
class JudgeStats:
    """Represent `JudgeStats` within this module.

The class groups the state and behavior required for JudgeStats."""
    model: str
    avg_score: float = 0.7
    avg_latency: float = 2.0
    avg_cost: float = 0.001
    consistency: float = 0.8
    fitness: float = 0.5


@dataclass
class SelectedJudge:
    """Represent `SelectedJudge` within this module.

The class groups the state and behavior required for SelectedJudge."""
    model: str
    weight: float


def _adaptive_threshold(values: Sequence[float], base: float) -> float:
    """Derive a selection threshold from recent judge fitness values."""
    if not values:
        return base
    median_val = statistics.median(values)
    return max(base, min(0.9, median_val * 0.6))


def _score_candidate(s: JudgeStats) -> float:
    """Compute a composite score used to rank candidate judge models."""
    qc = s.avg_score / max(s.avg_cost, 1e-6)
    qc_norm = min(10.0, 1.0 + qc ** 0.25)
    return max(0.0, W_FIT * s.fitness + W_QC * (qc_norm / 10.0))


def _choose_two(models: List[str], stats: Dict[str, JudgeStats]) -> List[SelectedJudge]:
    """Select two judges using fitness filtering and weighted randomization."""
    fitness_vals = [stats.get(m, JudgeStats(m)).fitness for m in models]
    thr = _adaptive_threshold(fitness_vals, MIN_FITNESS)
    valid = [m for m in models if stats.get(m, JudgeStats(m)).fitness >= thr]

    if len(valid) < 2:
        valid = models[:]

    if random.random() < EPSILON_RANDOM and len(valid) >= 2:
        picks = random.sample(valid, k=2)
        return [SelectedJudge(p, 1.0) for p in picks]

    scored = [(m, _score_candidate(stats.get(m, JudgeStats(m)))) for m in valid]
    total = sum(w for _, w in scored) or 1.0

    weights = [(m, w / total) for m, w in scored]

    def pick(wlist):
        """Draw one model from a normalized weight list."""
        r = random.random()
        acc = 0.0
        for name, w in wlist:
            acc += w
            if r <= acc:
                return name
        return wlist[-1][0]

    first = pick(weights)
    rest = [(m, w) for m, w in weights if m != first]
    second = pick(rest) if rest else first

    return [SelectedJudge(first, 1.0), SelectedJudge(second, 1.0)]
