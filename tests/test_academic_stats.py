# Objective: Tests for academic statistical utilities.
"""Tests for Holm-Bonferroni, Cohen's d, bootstrap CI, and kappa."""

from app.services.academic_stats import (
    anova_oneway,
    bootstrap_mean_ci,
    build_model_comparison_report,
    cohens_d,
    cohens_kappa,
    holm_bonferroni,
    kruskal_wallis,
    spearman,
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


def test_spearman_monotonic_positive():
    out = spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    assert out["method"] == "spearman"
    assert out["rho"] == 1.0


def test_spearman_insufficient_samples():
    out = spearman([1.0, 2.0], [3.0, 4.0])
    assert out["method"] == "insufficient_samples"
    assert out["rho"] is None


def test_kruskal_wallis_separates_groups():
    out = kruskal_wallis([1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0])
    assert out["method"] == "kruskal_wallis"
    assert out["n_groups"] == 3
    assert out["p_value"] < 0.05


def test_kruskal_wallis_needs_two_groups():
    out = kruskal_wallis([1.0, 2.0, 3.0])
    assert out["method"] == "insufficient_groups"


def test_anova_oneway_detects_mean_difference():
    out = anova_oneway([1.0, 1.1, 0.9], [5.0, 5.1, 4.9], [9.0, 9.1, 8.9])
    assert out["method"] == "anova_oneway"
    assert out["n_groups"] == 3
    assert out["p_value"] < 0.05


def test_anova_oneway_needs_two_groups():
    out = anova_oneway([1.0, 2.0])
    assert out["method"] == "insufficient_groups"
