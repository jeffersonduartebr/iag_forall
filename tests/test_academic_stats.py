# Objective: Tests for academic statistical utilities.
"""Tests for Holm-Bonferroni, Cohen's d, bootstrap CI, and kappa."""

from app.services.academic_stats import (
    bootstrap_mean_ci,
    build_model_comparison_report,
    cohens_d,
    cohens_kappa,
    holm_bonferroni,
)


def test_cohens_d_positive_difference():
    d = cohens_d([8.0, 8.5, 9.0], [5.0, 5.5, 6.0])
    assert d is not None
    assert d > 0


def test_bootstrap_mean_ci():
    ci = bootstrap_mean_ci([6.0, 7.0, 8.0, 7.5], n_bootstrap=500, seed=1)
    assert ci["mean"] == 7.125
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_holm_bonferroni_orders_by_pvalue():
    out = holm_bonferroni([("b", 0.04), ("a", 0.01)], alpha=0.05)
    assert out[0]["name"] == "a"
    assert out[0]["significant_holm"] is True


def test_cohens_kappa_perfect_agreement():
    k = cohens_kappa([2, 2, 3], [2, 2, 3])
    assert k["kappa"] == 1.0


def test_build_model_comparison_report():
    by_model = {
        "router": {"quality": [8.0, 7.5, 8.2], "latency": [1.0], "cost": [0.01]},
        "baseline": {"quality": [6.0, 6.5, 6.2], "latency": [0.5], "cost": [0.005]},
    }
    report = build_model_comparison_report(by_model)
    assert report["models"][0]["model"] == "router"
    assert len(report["comparisons"]) == 1
    assert report["comparisons"][0]["cohens_d"] is not None
