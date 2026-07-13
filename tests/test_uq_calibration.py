# Objective: Tests for UQ calibration metrics.
"""Tests for ECE and Spearman calibration helpers."""

from app.services.uq_calibration import (
    build_uq_calibration_report,
    expected_calibration_error,
    spearman_confidence_quality,
)


def test_expected_calibration_error_perfect():
    out = expected_calibration_error([0.9, 0.9, 0.1, 0.1], [1.0, 1.0, 0.0, 0.0], n_bins=2)
    assert out["ece"] is not None
    assert out["ece"] < 0.2


def test_spearman_confidence_quality():
    out = spearman_confidence_quality([0.2, 0.5, 0.8, 0.9], [4.0, 5.5, 7.0, 8.5])
    assert out["rho"] is not None
    assert out["rho"] > 0.5


def test_build_uq_calibration_report_per_theme():
    rows = [
        {
            "quality": 7.0,
            "metadata": {"confidence_score": 0.8, "benchmark_theme": "historia"},
        },
        {
            "quality": 5.0,
            "metadata": {"confidence_score": 0.4, "benchmark_theme": "historia"},
        },
        {
            "quality": 8.0,
            "metadata": {"confidence_score": 0.9, "benchmark_theme": "fisica"},
        },
    ]
    report = build_uq_calibration_report(rows)
    assert report["n_with_confidence"] == 3
    assert "historia" in report["per_theme"]
    assert "global" in report
