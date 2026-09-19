# Objective: Test coverage for the uncertainty threshold default and freezing.
"""tau_uq has one default (0.45) and online tuners leave it alone under frozen policy."""

from types import SimpleNamespace

import pytest

from app.config.constants import DEFAULT_UNCERTAINTY_THRESHOLD
from app.config.settings_catalog import SETTINGS_DEFAULTS
from app.services import frozen_policy, nsga_tuning


class _Settings:
    UQ_CALIBRATION_ENABLED = True

    def __init__(self, values):
        self.values = dict(values)
        self.writes = []

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value, actor="system"):
        self.writes.append((key, value, actor))
        self.values[key] = value


def test_single_default_threshold():
    assert DEFAULT_UNCERTAINTY_THRESHOLD == 0.45
    assert float(SETTINGS_DEFAULTS["UNCERTAINTY_THRESHOLD"]) == DEFAULT_UNCERTAINTY_THRESHOLD
    # Consultas só visuais recebem u = 0.7 e precisam ser tratadas como incertas.
    assert 0.7 > DEFAULT_UNCERTAINTY_THRESHOLD


def test_tuners_do_not_write_under_frozen_policy(monkeypatch):
    stub = _Settings({"UNCERTAINTY_THRESHOLD": "0.45"})
    monkeypatch.setattr(nsga_tuning, "settings", stub)
    monkeypatch.setattr(nsga_tuning, "is_frozen_policy_active", lambda: True)

    assert nsga_tuning.tune_uncertainty_threshold(current_efficiency=0.5) == 0.45
    assert nsga_tuning.calibrate_uncertainty_threshold() == {"status": "frozen"}
    assert stub.writes == []


def test_tuner_still_adjusts_when_not_frozen(monkeypatch):
    stub = _Settings({"UNCERTAINTY_THRESHOLD": "0.45"})
    monkeypatch.setattr(nsga_tuning, "settings", stub)
    monkeypatch.setattr(nsga_tuning, "is_frozen_policy_active", lambda: False)
    monkeypatch.setattr(nsga_tuning, "NSGA_UQ_THRESH", SimpleNamespace(set=lambda v: None))

    assert nsga_tuning.tune_uncertainty_threshold(current_efficiency=0.5) == 0.4
    assert stub.writes and stub.writes[0][0] == "UNCERTAINTY_THRESHOLD"


def test_frozen_snapshot_records_threshold(monkeypatch):
    import app.settings_dynamic as sd

    fake = SimpleNamespace(
        NSGA_W_QUALITY=1.0,
        NSGA_W_LATENCY=0.5,
        NSGA_W_COST=50.0,
        get=lambda key, default=None: {"UNCERTAINTY_THRESHOLD": "0.45"}.get(key, default),
    )
    monkeypatch.setattr(sd, "settings", fake)
    snapshot = frozen_policy.build_frozen_snapshot()
    assert snapshot["UNCERTAINTY_THRESHOLD"] == 0.45


def test_bucket_qualities_by_family_and_uncertainty():
    rows = [
        ("openai/gpt-5.5", 9.0, 0.9),  # SOTA, alta incerteza
        ("ollama/gemma3:4b", 4.0, 0.9),  # local, alta
        ("ollama/gemma3:4b", 8.0, 0.1),  # local, baixa
        ("openai/gpt-5.5", 7.0, 0.1),  # SOTA em baixa incerteza: ignorado
        ("ollama/qwen", None, None),  # defaults: qualidade 5, incerteza 0.5 (> 0.45)
    ]
    buckets = nsga_tuning.bucket_qualities(rows, 0.45)
    assert buckets == {"sota_high_uq": [9.0], "local_high_uq": [4.0, 5.0], "local_low_uq": [8.0]}


def test_propose_risk_factor_rules():
    sota, local_high, local_low = nsga_tuning.RISK_RULES
    few, low, high = [9.0] * 5, [4.0] * 20, [9.0] * 20
    assert nsga_tuning.propose_risk_factor(sota, 1.3, few, 0.1) is None  # amostras insuficientes
    assert nsga_tuning.propose_risk_factor(sota, 1.3, low, 0.1) == pytest.approx(1.2)
    assert nsga_tuning.propose_risk_factor(sota, 2.0, high, 0.1) is None  # já no teto
    assert nsga_tuning.propose_risk_factor(local_high, 0.35, low, 0.1) == pytest.approx(0.3)
    assert nsga_tuning.propose_risk_factor(local_low, 1.1, low, 0.1) is None  # só aumenta
    assert nsga_tuning.propose_risk_factor(local_low, 1.1, high, 0.1) == pytest.approx(1.2)


def test_tune_risk_factors_applies_rules(monkeypatch):
    stub = _Settings({"UNCERTAINTY_THRESHOLD": "0.45"})
    stub.RISK_FACTOR_ADAPT_ENABLED = True
    stub.RISK_FACTOR_ADAPT_RATE = 0.1
    stub.RISK_FACTOR_SOTA_HIGH_UQ = 1.3
    stub.RISK_FACTOR_LOCAL_HIGH_UQ = 0.6
    stub.RISK_FACTOR_LOCAL_LOW_UQ = 1.1
    rows = [("openai/gpt-5.5", 6.0, 0.9)] * 50 + [("ollama/gemma3:4b", 8.0, 0.1)] * 60
    monkeypatch.setattr(nsga_tuning, "settings", stub)
    monkeypatch.setattr(nsga_tuning, "_recent_quality_rows", lambda: rows)

    out = nsga_tuning.tune_risk_factors()
    assert out["status"] == "ok"
    assert out["adjustments"] == ["SOTA_HIGH: 1.30 -> 1.20", "LOCAL_LOW: 1.10 -> 1.20"]
    assert out["metrics"]["sota_high_uq_count"] == 50
    assert ("RISK_FACTOR_SOTA_HIGH_UQ", "1.2", "risk-tuner") in stub.writes

    monkeypatch.setattr(nsga_tuning, "_recent_quality_rows", lambda: rows[:10])
    assert nsga_tuning.tune_risk_factors() == {"status": "insufficient_data", "count": 10}
