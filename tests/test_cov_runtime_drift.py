# Objective: Drift detector persistence (baseline/stats in Redis), degraded Redis, drift events and resets.
"""Coverage of app.drift_detector state handling against fakeredis."""

from __future__ import annotations

import json
from types import SimpleNamespace

import app.drift_detector as dd
import pytest


@pytest.fixture
def make(monkeypatch, fake_redis):
    """Build a fresh (non-singleton-leaking) detector backed by fakeredis."""
    monkeypatch.setattr(dd, "settings", SimpleNamespace(DRIFT_WINDOW_SIZE=20, DRIFT_THRESHOLD=0.3))
    monkeypatch.setattr(dd, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(dd.QueryDriftDetector, "_instance", None)

    def _make():
        dd.QueryDriftDetector._instance = None
        return dd.get_drift_detector()

    _make.redis = fake_redis
    return _make


def _feed(det, vec, n):
    return [det.record_query(vec) for _ in range(n)][-1]


def test_baseline_is_persisted_once_ready_and_reloaded_after_restart(make):
    """Regression: saving only on count % 100 never fired, because the baseline stops growing at 50."""
    det = make()
    _feed(det, [1.0, 0.0], dd.BASELINE_READY_SAMPLES)
    stored = json.loads(make.redis.get(dd.REDIS_KEY_BASELINE_CENTROID))
    assert stored["sample_count"] == dd.BASELINE_READY_SAMPLES
    assert stored["centroid"] == [1.0, 0.0]

    restarted = make()
    assert restarted is not det
    assert restarted.get_status()["baseline_ready"] is True
    assert restarted.record_query([1.0, 0.0])["message"] == "Collecting samples"


def test_drift_detected_counts_events_and_persists_stats(make):
    det = make()
    _feed(det, [1.0, 0.0], dd.BASELINE_READY_SAMPLES)
    calm = _feed(det, [1.0, 0.0], 10)
    assert calm["drift_detected"] is False and calm["drift_score"] == pytest.approx(0.0)
    det._recent_embeddings.clear()
    shifted = _feed(det, [0.0, 1.0], 10)  # recent centroid orthogonal to baseline
    assert shifted["drift_detected"] is True
    assert shifted["drift_score"] == pytest.approx(1.0)
    json.dumps(shifted)  # regression: numpy.bool_/float64 leaked into the result and broke JSON responses
    status = det.get_status()
    assert status["drift_events"] == 1
    assert status["total_queries"] == 70
    assert make.redis.get(dd.REDIS_KEY_DRIFT_STATS) is None  # only every 50th query on the drift path
    _feed(det, [0.0, 1.0], 30)
    stats = json.loads(make.redis.get(dd.REDIS_KEY_DRIFT_STATS))
    assert (stats["total_queries"], stats["drift_events"]) == (100, 31)


def test_stats_are_restored_from_redis(make):
    make.redis.set(dd.REDIS_KEY_DRIFT_STATS, json.dumps({"total_queries": 40, "drift_events": 4}))
    status = make().get_status()
    assert (status["total_queries"], status["drift_events"], status["drift_rate"]) == (40, 4, 0.1)


def test_corrupted_redis_state_does_not_break_detector(make):
    make.redis.set(dd.REDIS_KEY_BASELINE_CENTROID, "{broken")
    det = make()
    assert det.get_status()["baseline_ready"] is False
    assert det.record_query([1.0])["message"] == "Building baseline"


def test_detector_without_redis_still_works(make, monkeypatch):
    monkeypatch.setattr(dd, "get_redis", lambda: None)
    det = make()
    _feed(det, [1.0, 1.0], dd.BASELINE_READY_SAMPLES)
    det._save_stats()
    det.reset_baseline()
    assert det.get_status()["baseline_samples"] == 0


def test_redis_write_failures_are_swallowed(make, monkeypatch):
    det = make()

    def _boom(*a, **k):
        raise ConnectionError("redis down")

    monkeypatch.setattr(make.redis, "set", _boom)
    monkeypatch.setattr(make.redis, "delete", _boom)
    _feed(det, [0.5, 0.5], dd.BASELINE_READY_SAMPLES)  # triggers _save_baseline
    det._save_stats()
    det.reset_baseline()
    assert det.get_status()["recent_samples"] == 0


def test_reset_then_force_baseline_update_from_recent_window(make):
    det = make()
    _feed(det, [1.0, 0.0], dd.BASELINE_READY_SAMPLES)
    det.reset_baseline()
    assert make.redis.get(dd.REDIS_KEY_BASELINE_CENTROID) is None
    for _ in range(12):
        det._recent_embeddings.append(dd.np.array([0.0, 2.0]))
    out = det.force_baseline_update()
    assert out == {"success": True, "baseline_samples": 12}
    assert json.loads(make.redis.get(dd.REDIS_KEY_BASELINE_CENTROID))["centroid"] == [0.0, 2.0]


def test_force_update_reports_centroid_failure(make, monkeypatch):
    det = make()
    for _ in range(10):
        det._recent_embeddings.append(dd.np.array([1.0]))
    monkeypatch.setattr(det, "_compute_centroid", lambda _e: None)
    assert det.force_baseline_update()["success"] is False


def test_check_drift_guards(make):
    det = make()
    assert det._check_drift()["message"] == "No baseline"
    det._baseline_centroid = dd.np.array([1.0])
    assert det._check_drift()["message"] == "No recent data"
    assert det._compute_centroid([]) is None
