# Objective: Academic statistical utilities for eval reports.
"""Holm-Bonferroni, effect sizes, bootstrap CI, and Cohen's kappa."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


def cohens_d(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Compute pooled Cohen's d between two samples."""
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) < 2 or len(y) < 2:
        return None
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap percentile CI for the sample mean."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    if arr.size == 1:
        val = float(arr[0])
        return {"mean": val, "ci_low": val, "ci_high": val, "n": 1}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(max(100, n_bootstrap)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    low_q = alpha / 2.0
    high_q = 1.0 - alpha / 2.0
    ci_low, ci_high = np.quantile(means, [low_q, high_q])
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n": int(arr.size),
        "alpha": alpha,
    }


def holm_bonferroni(p_values: Sequence[Tuple[str, float]], alpha: float = 0.05) -> List[Dict[str, Any]]:
    """Apply Holm-Bonferroni correction to named p-values."""
    items = [(name, float(p)) for name, p in p_values]
    if not items:
        return []
    m = len(items)
    sorted_items = sorted(items, key=lambda x: x[1])
    out: List[Dict[str, Any]] = []
    for rank, (name, p_raw) in enumerate(sorted_items, start=1):
        threshold = alpha / (m - rank + 1)
        out.append(
            {
                "name": name,
                "p_raw": p_raw,
                "p_holm_threshold": threshold,
                "significant_holm": p_raw <= threshold,
                "rank": rank,
            }
        )
    return out


def welch_ttest(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Welch t-test with scipy fallback."""
    x = list(a)
    y = list(b)
    if len(x) < 2 or len(y) < 2:
        return {"p_value": None, "method": "insufficient_samples"}
    if scipy_stats is None:
        return {"p_value": None, "method": "scipy_unavailable"}
    try:
        _t, p = scipy_stats.ttest_ind(np.asarray(x, dtype=float), np.asarray(y, dtype=float), equal_var=False)
        return {"p_value": float(p), "method": "welch_ttest"}
    except Exception:
        return {"p_value": None, "method": "ttest_failed"}


def cohens_kappa(rater_a: Sequence[int], rater_b: Sequence[int]) -> Dict[str, Any]:
    """Compute Cohen's kappa for two ordinal raters."""
    if len(rater_a) != len(rater_b) or not rater_a:
        return {"kappa": None, "n": 0, "method": "insufficient_pairs"}
    a = np.asarray(list(rater_a), dtype=int)
    b = np.asarray(list(rater_b), dtype=int)
    categories = sorted(set(a.tolist()) | set(b.tolist()))
    n = len(a)
    table = np.zeros((len(categories), len(categories)), dtype=float)
    cat_index = {c: i for i, c in enumerate(categories)}
    for av, bv in zip(a, b):
        table[cat_index[int(av)], cat_index[int(bv)]] += 1.0
    po = float(np.trace(table) / n)
    row_marginals = table.sum(axis=1) / n
    col_marginals = table.sum(axis=0) / n
    pe = float(np.sum(row_marginals * col_marginals))
    if pe >= 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1.0 - pe)
    return {"kappa": float(kappa), "n": n, "observed_agreement": po, "expected_agreement": pe, "method": "cohen_kappa"}


def build_model_comparison_report(
    by_model: Dict[str, Dict[str, List[float]]],
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Leaderboard with Holm-corrected comparisons against the top model."""
    leaderboard: List[Dict[str, Any]] = []
    for model, metrics in by_model.items():
        quality = metrics.get("quality") or []
        leaderboard.append(
            {
                "model": model,
                "n": len(quality),
                "quality_mean": float(np.mean(quality)) if quality else 0.0,
                "quality_ci": bootstrap_mean_ci(quality, alpha=alpha),
                "latency_mean": float(np.mean(metrics.get("latency") or [])) if metrics.get("latency") else 0.0,
                "cost_mean": float(np.mean(metrics.get("cost") or [])) if metrics.get("cost") else 0.0,
                "_quality_values": list(quality),
            }
        )
    leaderboard.sort(key=lambda item: item["quality_mean"], reverse=True)
    top = leaderboard[0] if leaderboard else None
    comparisons: List[Dict[str, Any]] = []
    holm_inputs: List[Tuple[str, float]] = []
    if top:
        for item in leaderboard[1:]:
            test = welch_ttest(top["_quality_values"], item["_quality_values"])
            d = cohens_d(top["_quality_values"], item["_quality_values"])
            p_val = test.get("p_value")
            comparisons.append(
                {
                    "top_model": top["model"],
                    "challenger_model": item["model"],
                    "delta_quality_mean": top["quality_mean"] - item["quality_mean"],
                    "cohens_d": d,
                    **test,
                }
            )
            if p_val is not None:
                holm_inputs.append((f"{top['model']} vs {item['model']}", float(p_val)))
    holm = holm_bonferroni(holm_inputs, alpha=alpha)
    for comp in comparisons:
        key = f"{comp['top_model']} vs {comp['challenger_model']}"
        comp["holm"] = next((h for h in holm if h["name"] == key), None)

    for item in leaderboard:
        item.pop("_quality_values", None)

    return {
        "models": leaderboard,
        "comparisons": comparisons,
        "holm_bonferroni": holm,
        "significance_threshold": alpha,
    }
