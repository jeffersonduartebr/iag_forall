# Objective: Uncertainty quantification calibration metrics for eval reports.
"""ECE and rank correlation for confidence vs quality."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def expected_calibration_error(
    confidences: Sequence[float],
    outcomes: Sequence[float],
    *,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute ECE using equal-width confidence bins."""
    conf = np.asarray(list(confidences), dtype=float)
    out = np.asarray(list(outcomes), dtype=float)
    if conf.size == 0:
        return {"ece": None, "n": 0, "bins": []}
    conf = np.clip(conf, 0.0, 1.0)
    bins: List[Dict[str, Any]] = []
    ece = 0.0
    for i in range(max(1, n_bins)):
        low = i / n_bins
        high = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf >= low) & (conf < high)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(conf[mask]))
        bin_acc = float(np.mean(out[mask]))
        weight = float(np.sum(mask) / conf.size)
        gap = abs(bin_acc - bin_conf)
        ece += weight * gap
        bins.append(
            {
                "bin": i,
                "low": low,
                "high": high,
                "count": int(np.sum(mask)),
                "mean_confidence": bin_conf,
                "mean_outcome": bin_acc,
                "gap": gap,
                "weight": weight,
            }
        )
    return {"ece": float(ece), "n": int(conf.size), "bins": bins, "n_bins": n_bins}


def spearman_confidence_quality(
    confidences: Sequence[float],
    qualities: Sequence[float],
) -> Dict[str, Any]:
    """Spearman correlation between confidence and judge quality.

    Thin wrapper over :func:`academic_stats.spearman` (single source of truth).
    """
    from .academic_stats import spearman

    return spearman(confidences, qualities)


def build_uq_calibration_report(rows: List[Dict[str, Any]], *, quality_threshold: float = 6.0) -> Dict[str, Any]:
    """Aggregate UQ calibration metrics globally and per benchmark theme."""
    confidences: List[float] = []
    qualities: List[float] = []
    outcomes: List[float] = []
    by_theme: Dict[str, Dict[str, List[float]]] = {}

    for row in rows:
        meta = row.get("metadata") or {}
        conf = meta.get("confidence_score")
        if conf is None:
            continue
        quality = float(row.get("quality", 0.0) or 0.0)
        conf_f = float(conf)
        if conf_f > 1.0:
            conf_f = conf_f / 10.0
        conf_f = max(0.0, min(1.0, conf_f))
        confidences.append(conf_f)
        qualities.append(quality)
        outcomes.append(1.0 if quality >= quality_threshold else 0.0)
        theme = str(meta.get("benchmark_theme") or "unknown")
        slot = by_theme.setdefault(theme, {"confidences": [], "qualities": [], "outcomes": []})
        slot["confidences"].append(conf_f)
        slot["qualities"].append(quality)
        slot["outcomes"].append(1.0 if quality >= quality_threshold else 0.0)

    per_theme: Dict[str, Any] = {}
    for theme, bucket in by_theme.items():
        per_theme[theme] = {
            "ece": expected_calibration_error(bucket["confidences"], bucket["outcomes"]),
            "spearman": spearman_confidence_quality(bucket["confidences"], bucket["qualities"]),
        }

    return {
        "n_with_confidence": len(confidences),
        "quality_threshold": quality_threshold,
        "global": {
            "ece": expected_calibration_error(confidences, outcomes),
            "spearman": spearman_confidence_quality(confidences, qualities),
        },
        "per_theme": per_theme,
    }
