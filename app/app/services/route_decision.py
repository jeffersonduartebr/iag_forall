# Objective: Make one routing decision reconstructible after the fact.
"""The candidates considered, their objective vectors, and the Pareto front.

Both ``candidates`` and ``pareto_front`` used to be written to the response as
literal empty lists, in both routing paths. The frente de Pareto that names the
central contribution of this work never reached disk, and there was no way to
answer "why was this model chosen?" from the stored data — only "which one".

What is recorded here is the state the scorer already had in memory:

- ``quality`` — the Thompson sample from the shared posterior, before the risk
  factor, on the 0-10 scale;
- ``latency_s`` and ``cost_usd`` — the shared EMA estimates, or the heuristic
  used while a model has too few observations;
- ``risk`` — the multiplier applied for locality and uncertainty;
- ``score`` — the scalarised ``model_score`` under the NSGA-II weights, which
  is what the ranking actually used.

The Pareto front is computed over the three raw objectives, *not* over the
scalarised score. A weighted sum can only ever return one point of the front,
so recording the front separately is what distinguishes "this was the best
trade-off under today's weights" from "this was the only non-dominated option".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class Candidate:
    """One model as the scorer saw it, with its three objectives."""

    model: str
    quality: float
    latency_s: float
    cost_usd: float
    risk: float
    score: float

    def objectives(self) -> tuple:
        """(maximise, minimise, minimise) — the vector the front is computed on."""
        return (self.quality, self.latency_s, self.cost_usd)


def _dominates(a: Candidate, b: Candidate) -> bool:
    """True when ``a`` is at least as good on every objective and better on one.

    Quality is maximised; latency and cost are minimised.
    """
    at_least_as_good = (
        a.quality >= b.quality and a.latency_s <= b.latency_s and a.cost_usd <= b.cost_usd
    )
    strictly_better = (
        a.quality > b.quality or a.latency_s < b.latency_s or a.cost_usd < b.cost_usd
    )
    return at_least_as_good and strictly_better


def pareto_front(candidates: Sequence[Candidate]) -> List[Candidate]:
    """The non-dominated subset, in the order given.

    O(n²) is deliberate: the candidate list is the configured model roster, a
    handful of entries, and a clear definition is worth more here than an
    asymptotically better algorithm nobody will read.
    """
    return [c for c in candidates if not any(_dominates(other, c) for other in candidates if other is not c)]


def decision_record(
    candidates: Sequence[Candidate],
    *,
    chosen: str,
    top2: Sequence[str],
    weights: Dict[str, float],
    uncertainty: float,
) -> Dict[str, Any]:
    """The JSON-serialisable record of one routing decision.

    ``weights`` are included because the scalarised score is meaningless
    without them, and they move: the NSGA-II updater rewrites them in the
    background. A row stored without the weights that produced it cannot be
    re-derived later.
    """
    front = {c.model for c in pareto_front(candidates)}
    return {
        "chosen": chosen,
        "top2": list(top2),
        "uncertainty": round(float(uncertainty), 4),
        "weights": {k: float(v) for k, v in (weights or {}).items()},
        "candidates": [
            {**asdict(c), "on_pareto_front": c.model in front} for c in candidates
        ],
        "pareto_front": sorted(front),
    }
