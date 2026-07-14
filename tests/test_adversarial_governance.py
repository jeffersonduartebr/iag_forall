# Objective: Tests for closed-loop adversarial governance (roadmap #17).
"""Cluster risk memory, online-loop closure wiring, and risk-based escalation."""

import json

import app.services.adversarial_governance as ag


class _FakeSettings:
    def __init__(self, values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


def _enable(monkeypatch, **overrides):
    """Enable governance with test-friendly thresholds; force in-memory store."""
    values = {
        "ADVGOV_ENABLED": "1",
        "ADVGOV_FAIL_SCORE_THRESHOLD": "7.0",
        "ADVGOV_CLUSTER_MIN_SAMPLES": "3",
        "ADVGOV_CLUSTER_FAILURE_RATE_THRESHOLD": "0.5",
        "ADVGOV_CLUSTER_TTL_S": "3600",
        "ADVGOV_ESCALATION_ENABLED": "1",
        "ADVGOV_ESCALATION_MODELS": "[]",
        "UNCERTAINTY_THRESHOLD": "0.7",
    }
    values.update(overrides)
    monkeypatch.setattr(ag, "_settings", lambda: _FakeSettings(values))
    monkeypatch.setattr(ag, "_read_redis", lambda: None)
    ag.reset_state()


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #
def test_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(ag, "_settings", lambda: _FakeSettings({"ADVGOV_ENABLED": "0"}))
    ag.reset_state()
    assert ag.record_adversarial_outcome(cluster_id="x", model="m", score=1.0) is None
    assert ag.suggest_escalation(cluster_id="x", candidate_models=["openai/gpt"], uncertainty=0.99) is None


# --------------------------------------------------------------------------- #
# Cluster risk memory
# --------------------------------------------------------------------------- #
def test_cluster_high_risk_after_repeated_failures(monkeypatch):
    _enable(monkeypatch)
    for _ in range(3):
        ag.record_adversarial_outcome(cluster_id="calculo", model="m", score=2.0, learn=False)
    risk = ag.get_cluster_risk("calculo")
    assert risk["n"] == 3
    assert risk["failures"] == 3
    assert risk["failure_rate"] == 1.0
    assert risk["mean_score"] == 2.0
    assert risk["high_risk"] is True


def test_cluster_not_high_risk_when_answers_pass(monkeypatch):
    _enable(monkeypatch)
    for _ in range(5):
        ag.record_adversarial_outcome(cluster_id="historia", model="m", score=9.0, learn=False)
    risk = ag.get_cluster_risk("historia")
    assert risk["failure_rate"] == 0.0
    assert risk["high_risk"] is False


def test_cluster_below_min_samples_not_flagged(monkeypatch):
    _enable(monkeypatch, ADVGOV_CLUSTER_MIN_SAMPLES="5")
    for _ in range(2):
        ag.record_adversarial_outcome(cluster_id="fisica", model="m", score=1.0, learn=False)
    risk = ag.get_cluster_risk("fisica")
    assert risk["failure_rate"] == 1.0
    assert risk["high_risk"] is False  # not enough samples yet


def test_unknown_cluster_is_safe(monkeypatch):
    _enable(monkeypatch)
    risk = ag.get_cluster_risk("never-seen")
    assert risk["n"] == 0
    assert risk["high_risk"] is False
    assert risk["mean_score"] is None


# --------------------------------------------------------------------------- #
# Online-loop closure wiring
# --------------------------------------------------------------------------- #
def test_record_triggers_online_loop_with_failure_flag(monkeypatch):
    _enable(monkeypatch)
    calls = []
    monkeypatch.setattr(ag, "_close_online_loop", lambda **kw: calls.append(kw))
    risk = ag.record_adversarial_outcome(cluster_id="quimica", model="ollama/x", score=3.0, query="q")
    assert len(calls) == 1
    assert calls[0]["is_failure"] is True
    assert calls[0]["model"] == "ollama/x"
    assert risk["n"] == 1 and risk["failures"] == 1


def test_passing_score_is_not_a_failure(monkeypatch):
    _enable(monkeypatch)
    calls = []
    monkeypatch.setattr(ag, "_close_online_loop", lambda **kw: calls.append(kw))
    ag.record_adversarial_outcome(cluster_id="quimica", model="m", score=8.5, query="q")
    assert calls[0]["is_failure"] is False


def test_learn_false_skips_online_loop(monkeypatch):
    _enable(monkeypatch)
    calls = []
    monkeypatch.setattr(ag, "_close_online_loop", lambda **kw: calls.append(kw))
    ag.record_adversarial_outcome(cluster_id="quimica", model="m", score=2.0, learn=False)
    assert calls == []


# --------------------------------------------------------------------------- #
# Risk-based escalation
# --------------------------------------------------------------------------- #
def test_escalation_triggered_by_high_uncertainty(monkeypatch):
    _enable(monkeypatch)
    out = ag.suggest_escalation(
        cluster_id=None,
        candidate_models=["ollama/local", "openai/gpt-5.2"],
        current_model="ollama/local",
        uncertainty=0.9,
    )
    assert out == "openai/gpt-5.2"


def test_no_escalation_when_calm(monkeypatch):
    _enable(monkeypatch)
    out = ag.suggest_escalation(
        cluster_id=None,
        candidate_models=["ollama/local", "openai/gpt-5.2"],
        current_model="ollama/local",
        uncertainty=0.1,
    )
    assert out is None


def test_escalation_triggered_by_cluster_risk(monkeypatch):
    _enable(monkeypatch)
    for _ in range(3):
        ag.record_adversarial_outcome(cluster_id="fragil", model="m", score=1.0, learn=False)
    out = ag.suggest_escalation(
        cluster_id="fragil",
        candidate_models=["ollama/local", "anthropic/claude-opus-4-8"],
        current_model="ollama/local",
        uncertainty=0.0,
    )
    assert out == "anthropic/claude-opus-4-8"


def test_escalation_prefers_configured_target(monkeypatch):
    _enable(monkeypatch, ADVGOV_ESCALATION_MODELS=json.dumps(["openai/gpt-5.2"]))
    out = ag.suggest_escalation(
        cluster_id=None,
        candidate_models=["ollama/local", "anthropic/claude-opus-4-8", "openai/gpt-5.2"],
        current_model="ollama/local",
        uncertainty=0.9,
    )
    assert out == "openai/gpt-5.2"


def test_no_escalation_when_only_current_candidate(monkeypatch):
    _enable(monkeypatch)
    out = ag.suggest_escalation(
        cluster_id=None,
        candidate_models=["openai/gpt-5.2"],
        current_model="openai/gpt-5.2",
        uncertainty=0.99,
    )
    assert out is None


def test_escalation_respects_disable_flag(monkeypatch):
    _enable(monkeypatch, ADVGOV_ESCALATION_ENABLED="0")
    out = ag.suggest_escalation(
        cluster_id=None,
        candidate_models=["ollama/local", "openai/gpt-5.2"],
        current_model="ollama/local",
        uncertainty=0.99,
    )
    assert out is None
