# Objective: Test coverage for the NSGA-II online tuners (global weights, UQ calibration, candidates).
"""services.nsga_tuning controllers and nsga_weights_updater.load_candidate_models."""

import json
from types import SimpleNamespace

import fakeredis
import pytest
from app.services import nsga_tuning

from app import nsga_weights_updater as nwu


class _Settings:
    NSGA_W_QUALITY = 1.0
    NSGA_W_LATENCY = 0.5
    NSGA_W_COST = 50.0
    UQ_CALIBRATION_ENABLED = True
    UQ_QUALITY_GAP_RELAX = 0.5
    UQ_QUALITY_GAP_TIGHTEN = 2.0

    def __init__(self, **values):
        self.values = {"UNCERTAINTY_THRESHOLD": "0.45", **values}
        self.writes = []

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value, actor="system"):
        self.writes.append((key, value, actor))


@pytest.fixture
def stub(monkeypatch):
    settings = _Settings()
    monkeypatch.setattr(nsga_tuning, "settings", settings)
    monkeypatch.setattr(nsga_tuning, "is_frozen_policy_active", lambda: False)
    monkeypatch.setattr(nsga_tuning, "NSGA_UQ_THRESH", SimpleNamespace(set=lambda v: None))
    return settings


# ---------------------------------------------------------------- pesos globais


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ((4.0, 0.0, 8.0), {"NSGA_W_LATENCY": (0.5, 0.6)}),  # lento
        ((0.5, 0.0, 8.0), {"NSGA_W_LATENCY": (0.5, 0.45)}),  # rápido: relaxa latência
        ((2.0, 0.0, 6.0), {"NSGA_W_QUALITY": (1.0, 1.2)}),  # qualidade baixa
        ((2.0, 0.02, 8.0), {"NSGA_W_COST": (50.0, 55.0)}),  # caro
        ((2.0, 0.005, 8.0), {}),  # dentro das metas
    ],
)
def test_propose_strategy_weights(metrics, expected):
    got = nsga_tuning.propose_strategy_weights(metrics, 1.0, 0.5, 50.0)
    assert {k: pytest.approx(v) for k, v in got.items()} == expected


def test_propose_strategy_weights_respects_bounds():
    assert nsga_tuning.propose_strategy_weights((9.0, 1.0, 1.0), 5.0, 2.0, 100.0) == {}
    assert nsga_tuning.propose_strategy_weights((0.1, 0.0, 9.0), 1.0, 0.1, 50.0) == {}


def test_tune_global_strategy_weights_writes_in_order(stub):
    nsga_tuning.tune_global_strategy_weights((4.0, 0.02, 6.0))
    assert stub.writes == [
        ("NSGA_W_LATENCY", "0.6", "nsga-updater"),
        ("NSGA_W_QUALITY", "1.2", "nsga-updater"),
        ("NSGA_W_COST", "55.0", "nsga-updater"),
    ]


def test_tune_uncertainty_threshold_branches(stub):
    assert nsga_tuning.tune_uncertainty_threshold(current_efficiency=5.0) == pytest.approx(0.5)
    assert nsga_tuning.tune_uncertainty_threshold(current_efficiency=3.0) == pytest.approx(0.45)
    assert [w[0] for w in stub.writes] == ["UNCERTAINTY_THRESHOLD"]  # KEEP não grava


# ---------------------------------------------------------------- calibração de u(q)


def test_split_by_uncertainty_uses_defaults_for_missing_values():
    rows = [("m", 9.0, 0.9), ("m", 3.0, 0.1), ("m", None, None), ("m", 7.0, "0.46")]
    assert nsga_tuning.split_by_uncertainty(rows, 0.45) == ([9.0, 5.0, 7.0], [3.0])


@pytest.mark.parametrize(
    ("gap", "expected"),
    [(0.1, (0.5, "RELAX")), (3.0, (0.4, "TIGHTEN")), (1.0, (0.45, "KEEP"))],
)
def test_decide_threshold(gap, expected):
    new, action = nsga_tuning.decide_threshold(0.45, gap, 0.5, 2.0)
    assert (round(new, 2), action) == expected


def test_decide_threshold_bounds():
    assert nsga_tuning.decide_threshold(0.80, 0.0, 0.5, 2.0) == (0.80, "RELAX")
    assert nsga_tuning.decide_threshold(0.20, 9.0, 0.5, 2.0) == (0.20, "TIGHTEN")


def _rows(high_quality, low_quality, n=60):
    return [("m", high_quality, 0.9)] * n + [("m", low_quality, 0.1)] * n


def test_calibrate_tightens_when_high_uq_is_much_worse(monkeypatch, stub):
    monkeypatch.setattr(nsga_tuning, "_recent_quality_rows", lambda: _rows(4.0, 8.0))
    out = nsga_tuning.calibrate_uncertainty_threshold()
    assert (out["status"], out["action"], out["old_threshold"]) == ("ok", "TIGHTEN", 0.45)
    assert out["new_threshold"] == pytest.approx(0.40)
    assert out["metrics"]["quality_gap"] == pytest.approx(4.0)
    assert stub.writes == [("UNCERTAINTY_THRESHOLD", "0.4", "uq-calibrator")]


def test_calibrate_keeps_threshold_inside_band(monkeypatch, stub):
    monkeypatch.setattr(nsga_tuning, "_recent_quality_rows", lambda: _rows(7.0, 8.0))
    out = nsga_tuning.calibrate_uncertainty_threshold()
    assert (out["action"], out["new_threshold"]) == ("KEEP", 0.45)
    assert stub.writes == []


def test_calibrate_guards(monkeypatch, stub):
    monkeypatch.setattr(nsga_tuning, "_recent_quality_rows", lambda: _rows(4.0, 8.0, n=10))
    assert nsga_tuning.calibrate_uncertainty_threshold() == {"status": "insufficient_data", "count": 20}

    monkeypatch.setattr(nsga_tuning, "_recent_quality_rows", lambda: [("m", 8.0, 0.1)] * 120)
    assert nsga_tuning.calibrate_uncertainty_threshold() == {
        "status": "insufficient_split",
        "high_count": 0,
        "low_count": 120,
    }

    def broken():
        raise RuntimeError("db down")

    monkeypatch.setattr(nsga_tuning, "_recent_quality_rows", broken)
    assert nsga_tuning.calibrate_uncertainty_threshold() == {"status": "error", "error": "db down"}

    stub.UQ_CALIBRATION_ENABLED = False
    assert nsga_tuning.calibrate_uncertainty_threshold() == {"status": "disabled"}


# ---------------------------------------------------------------- candidatos


@pytest.fixture
def candidates_env(monkeypatch):
    rds = fakeredis.FakeRedis()
    settings = SimpleNamespace(
        CANDIDATE_MODELS_LIST=["b", "a", "b", "", "c"],
        CANDIDATE_VISION_MODELS_LIST=[],
        CANDIDATE_MULTIMODAL_MODELS_LIST=["mm"],
    )
    monkeypatch.setattr(nwu, "redis_client", rds)
    monkeypatch.setattr(nwu, "settings", settings)
    return rds


def test_candidates_from_settings_keep_order_without_duplicates(candidates_env):
    assert nwu.load_candidate_models("text") == ["b", "a", "c"]
    assert nwu.load_candidate_models("multimodal") == ["mm"]


def test_candidates_redis_override(candidates_env):
    candidates_env.set(nwu.REDIS_KEY_CANDIDATES["text"], json.dumps(["x", 1]))
    assert nwu.load_candidate_models("text") == ["x", "1"]
    candidates_env.set(nwu.REDIS_KEY_CANDIDATES["text"], "{not json")
    assert nwu.load_candidate_models("text") == ["b", "a", "c"]


def test_candidates_fallback(candidates_env, monkeypatch):
    assert nwu.load_candidate_models("vision") == ["ollama/llava:7b", "ollama/moondream:latest"]
    assert nwu.load_candidate_models("audio") == []
    monkeypatch.setattr(nwu, "redis_client", None)
    assert nwu.load_candidate_models("text") == ["b", "a", "c"]
