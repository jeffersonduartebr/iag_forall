# Objective: Tests for experiment manifest and frozen policy helpers.
"""Tests for reproducibility manifest and frozen policy guards."""

from app.services.experiment_manifest import build_experiment_manifest
from app.services.frozen_policy import should_skip_eval_feedback


def test_build_experiment_manifest_has_core_fields(monkeypatch):
    monkeypatch.setattr(
        "app.services.experiment_manifest._config_snapshot",
        lambda: {"NSGA_W_QUALITY": 0.4, "BANDIT_EPSILON": 0.1},
    )
    manifest = build_experiment_manifest(
        run_id="eval_test_1",
        benchmark_seed=42,
        benchmark_split="held_out",
        frozen_policy=True,
        benchmark_themes=["historia"],
    )
    assert manifest["run_id"] == "eval_test_1"
    assert manifest["frozen_policy"] is True
    assert manifest["benchmark"]["split_filter"] == "held_out"
    assert "config_snapshot" in manifest
    assert "catalog_fingerprint_sha256" in manifest["benchmark"]


def test_should_skip_eval_feedback_when_frozen():
    assert should_skip_eval_feedback({"frozen_policy": True}) is True
    assert should_skip_eval_feedback({"experiment_manifest": {"frozen_policy": True}}) is True
    assert should_skip_eval_feedback({"frozen_policy": False}) is False
