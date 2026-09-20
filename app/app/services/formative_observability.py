# Objective: Routing confusion matrix — detected complexity x model class x outcome verdict.
"""Was the request sent to the right class of model?

Two failures matter and they pull in opposite directions. Sending a simple
request to an expensive frontier model is **waste**: the answer is fine and the
money was not needed. Sending a complex request to a small local model is
**collapse**: the money was saved and the answer is not usable. A router that
only reports average quality and average cost hides both, because each one
improves one of the two averages.

The two axes are independent by construction. Complexity comes from
``services.query_complexity.detect_query_complexity``, computed from the
statement *before* any model is chosen; quality comes from the judge, *after*
the answer exists. Neither can be derived from the other, so the cell an item
lands in is evidence rather than tautology.

**Why ``quality_source`` is a required label.** Roughly 95% of traffic is not
judged (``JUDGE_MIN_SAMPLE_RATE``) and falls back to ``proxy_quality``, which is
the bandit's own posterior mean rescaled. A matrix that mixed those rows in
would be measuring the router against its own opinion of itself. The label keeps
them separable, and every report filters on ``judge``.

Two coverage caveats belong with any percentage computed from this metric:
``persist_log`` does not run for tool-call turns or for semantic cache hits, so
the denominator is not total traffic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from prometheus_client import Counter, Histogram

from app.observability import registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

#: Below this, an answer is not usable. Same threshold ROI analysis already uses
#: for "acceptable", so the two reports cannot disagree about what counts as bad.
QUALITY_FLOOR = 6.0

#: The complexities for which a small local model is the wrong tool.
DEMANDING = ("high", "expert")

#: The complexities that do not justify a frontier model.
UNDEMANDING = ("simple", "moderate")

MODEL_CLASSES = ("local", "cloud", "sota")
VERDICTS = ("ok", "waste", "collapse", "inconclusive")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

#: Model *class*, never the model name: with four complexities, three classes and
#: four verdicts this is 240 series at most, and it stays that way when the
#: candidate list changes.
ROUTER_ROUTING_MATRIX = Counter(
    "router_routing_matrix_total",
    "Detected complexity x model class x routing verdict",
    ["detected_complexity", "model_class", "verdict", "quality_source"],
    registry=registry,
)

ROUTER_MATRIX_QUALITY = Histogram(
    "router_matrix_quality",
    "Judged quality per complexity and model class",
    ["detected_complexity", "model_class"],
    buckets=(0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0),
    registry=registry,
)


def classify_model(model: str) -> str:
    """``local``, ``sota`` or ``cloud`` for one model name.

    Reuses the two predicates the router already decides with, so the matrix
    cannot disagree with the routing it is describing.
    """
    try:
        from app.utils.pricing import is_local_model

        if is_local_model(model):
            return "local"
    except Exception:  # pragma: no cover - defensive, metrics must never raise
        if "ollama" in (model or "").lower():
            return "local"
    try:
        from app.router_strategy import _is_sota

        if _is_sota(model):
            return "sota"
    except Exception:  # pragma: no cover
        pass
    return "cloud"


def routing_verdict(
    complexity: Optional[str],
    model_class: str,
    quality: float,
    quality_source: str,
) -> str:
    """Which cell of the matrix this request belongs in.

    ``waste`` is deliberately named a *candidate*, not a proof. That a cheaper
    model would have answered as well is counterfactual and cannot be observed
    on a single row; the offline report settles it by pairing against the local
    model's own distribution at the same complexity. Saying so is the difference
    between a number and a defensible number.
    """
    if quality_source != "judge" or not complexity:
        return "inconclusive"
    if model_class == "local" and complexity in DEMANDING and quality < QUALITY_FLOOR:
        return "collapse"
    if model_class in ("cloud", "sota") and complexity in UNDEMANDING and quality >= QUALITY_FLOOR:
        return "waste"
    return "ok"


def record_routing_cell(
    model: str,
    complexity: Optional[str],
    quality: float,
    quality_source: str,
) -> Dict[str, Any]:
    """Emit one observation. Returns the cell, for tests and for logging."""
    model_class = classify_model(model)
    verdict = routing_verdict(complexity, model_class, quality, quality_source)
    label_complexity = complexity or "unknown"

    ROUTER_ROUTING_MATRIX.labels(
        detected_complexity=label_complexity,
        model_class=model_class,
        verdict=verdict,
        quality_source=quality_source,
    ).inc()

    if quality_source == "judge":
        ROUTER_MATRIX_QUALITY.labels(
            detected_complexity=label_complexity, model_class=model_class
        ).observe(float(quality))

    return {
        "detected_complexity": label_complexity,
        "model_class": model_class,
        "verdict": verdict,
        "quality_source": quality_source,
        "quality": float(quality),
    }
