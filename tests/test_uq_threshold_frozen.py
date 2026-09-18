# Objective: Test coverage for the uncertainty threshold default and freezing.
"""tau_uq has one default (0.45) and online tuners leave it alone under frozen policy."""

from types import SimpleNamespace

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
