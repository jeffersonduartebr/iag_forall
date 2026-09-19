# Objective: Unit tests for the NSGA-II core and the calibration cycle.
"""nsga_core (portfolio search, convergence) and nsga_calibration (isolated steps)."""

import random

import pytest

from app import nsga_calibration, nsga_core


def _metrics():
    return {
        "fast_cheap": {"latency": 0.5, "cost": 0.0001, "quality": 6.0, "alignment": 1.0},
        "slow_good": {"latency": 8.0, "cost": 0.01, "quality": 9.0, "alignment": 1.0},
    }


def test_run_nsga_optimization_returns_normalized_portfolio():
    random.seed(7)
    weights, efficiency, (lat, cost, quality) = nsga_core.run_nsga_optimization(
        "text", ["fast_cheap", "slow_good"], _metrics(), n_pop=12, n_gen=4
    )
    assert set(weights) == {"fast_cheap", "slow_good"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert 0.5 <= lat <= 8.0 and 6.0 <= quality <= 9.0 and efficiency == pytest.approx(quality / max(0.01, lat))


def test_run_nsga_optimization_degenerate_inputs():
    assert nsga_core.run_nsga_optimization("text", [], {}) == ({}, 0.0, (0, 0, 0))
    only = nsga_core.run_nsga_optimization("text", ["fast_cheap"], _metrics())
    assert only == ({"fast_cheap": 1.0}, 1.0, (0.5, 0.0001, 6.0))


def test_convergence_metrics_short_and_stable_history():
    assert nsga_core.compute_convergence_metrics([1.0, 2.0]) == {"trend": 0.0, "variance": 0.0, "health": 1.0}
    stable = nsga_core.compute_convergence_metrics([3.0, 3.0, 3.0, 3.0])
    assert stable["variance"] == pytest.approx(0.0) and stable["health"] > 0.5


def test_calibration_cycle_isolates_failing_steps(monkeypatch):
    ran = []

    def boom():
        ran.append("boom")
        raise RuntimeError("step down")

    steps = (("first", boom), ("second", lambda: ran.append("second")))
    monkeypatch.setattr(nsga_calibration, "CALIBRATION_STEPS", steps)
    nsga_calibration.run_calibration_cycle()
    assert ran == ["boom", "second"]
