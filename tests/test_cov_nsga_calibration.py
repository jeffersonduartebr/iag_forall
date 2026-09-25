# Objective: Coverage for the NSGA-II worker's periodic calibration steps and status payload.
"""nsga_calibration: each step's wiring, predictor gauges, and the optional sections of the status payload."""

import sys
from types import SimpleNamespace

import pytest

from app import nsga_calibration as cal


def _raise(*_a, **_k):
    raise RuntimeError("unavailable")


@pytest.fixture
def modules(monkeypatch):
    """Light stand-ins for the lazily imported modules (predictor, cache, judges)."""
    calls = []
    fakes = {
        "app.online_predictor": SimpleNamespace(
            calibrate_all_predictors=lambda: calls.append("calibrate"),
            get_all_predictor_metrics=lambda: {"m1": {"brier_score": 0.1, "accuracy": 0.8}},
        ),
        "app.semantic_cache": SimpleNamespace(get_l1_cache_stats=lambda: {"size": 3}, get_cache_hit_rate=lambda: 0.4),
        "app.judges": SimpleNamespace(get_judge_calibration_metrics=lambda: {"j1": {"agreement": 0.9}}),
        "app.services.judge_calibration": SimpleNamespace(calibrate_judges=lambda: {"status": "calibrated"}),
    }
    for name, module in fakes.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls, fakes


def test_risk_and_uncertainty_steps_delegate_to_tuning(monkeypatch, caplog):
    monkeypatch.setattr(cal, "tune_risk_factors", lambda: {"status": "tuned"})
    monkeypatch.setattr(cal, "calibrate_uncertainty_threshold", lambda: {})
    with caplog.at_level("INFO", logger="nsga-updater"):
        cal._calibrate_risk_factors()
        cal._calibrate_uncertainty()
    assert "Risk factors: tuned" in caplog.text and "UQ threshold: unknown" in caplog.text


def test_predictor_step_calibrates_and_publishes_gauges(modules):
    from app.observability import PREDICTOR_ACCURACY, PREDICTOR_BRIER_SCORE, PREDICTOR_CALIBRATION_TEMP

    calls, _ = modules
    cal._calibrate_predictors()
    assert calls == ["calibrate"]
    assert PREDICTOR_BRIER_SCORE.labels(model="m1")._value.get() == pytest.approx(0.1)
    assert PREDICTOR_ACCURACY.labels(model="m1")._value.get() == pytest.approx(0.8)
    assert PREDICTOR_CALIBRATION_TEMP.labels(model="m1")._value.get() == pytest.approx(1.0)  # default


def test_publish_predictor_metrics_ignores_bad_payloads():
    cal._publish_predictor_metrics({"m1": None})  # None.get levanta; o erro é engolido


def test_judge_step_uses_db_only_module(modules, caplog):
    with caplog.at_level("INFO", logger="nsga-updater"):
        cal._calibrate_judges()
    assert "Judge calibration: calibrated" in caplog.text


def test_full_cycle_runs_real_steps(monkeypatch, modules):
    monkeypatch.setattr(cal, "tune_risk_factors", _raise)
    monkeypatch.setattr(cal, "calibrate_uncertainty_threshold", lambda: {"status": "ok"})
    cal.run_calibration_cycle()
    assert modules[0] == ["calibrate"]  # o passo de risco falhou e os demais rodaram


def _defaults(monkeypatch):
    monkeypatch.setattr(cal.settings, "get", lambda key, fallback=None: fallback)


def test_status_payload_includes_optional_metrics(monkeypatch, modules):
    _defaults(monkeypatch)
    payload = cal.calibration_status_payload()
    assert payload["predictor"]["models"] == {"m1": {"brier_score": 0.1, "accuracy": 0.8}}
    assert payload["cache"]["l1_stats"] == {"size": 3} and payload["cache"]["hit_rate"] == 0.4
    assert payload["judge"]["models"] == {"j1": {"agreement": 0.9}}
    assert payload["uncertainty"]["threshold"] == pytest.approx(cal.DEFAULT_UNCERTAINTY_THRESHOLD)
    assert payload["cache"]["threshold"] == pytest.approx(0.92)
    assert payload["risk_factors"]["sota_high_uq"] == pytest.approx(1.3)


def test_status_payload_omits_unavailable_sections(monkeypatch, modules):
    _defaults(monkeypatch)
    _, fakes = modules
    fakes["app.online_predictor"].get_all_predictor_metrics = _raise
    fakes["app.semantic_cache"].get_l1_cache_stats = _raise
    fakes["app.semantic_cache"].get_cache_hit_rate = _raise
    fakes["app.judges"].get_judge_calibration_metrics = _raise
    payload = cal.calibration_status_payload()
    assert "models" not in payload["predictor"] and "models" not in payload["judge"]
    assert "l1_stats" not in payload["cache"] and "hit_rate" not in payload["cache"]
