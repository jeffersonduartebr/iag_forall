# Objective: Tests for eval-driven NSGA/bandit feedback loop.
"""Tests for eval feedback integration."""

from __future__ import annotations


def test_apply_eval_run_feedback_boosts_quality_weight_on_low_scores(monkeypatch):
    from app.services import eval_feedback as ef

    updates = []

    class _Settings:
        NSGA_W_QUALITY = 1.0
        NSGA_W_LATENCY = 0.5
        NSGA_W_COST = 100.0

        def get(self, key, default=None):
            return {
                "EVAL_FEEDBACK_QUALITY_MIN": "6.0",
                "EVAL_FEEDBACK_QUALITY_MAX": "8.5",
                "BANDIT_EPSILON": "0.10",
                "EVAL_FEEDBACK_BANDIT_EPSILON_MAX": "0.35",
                "EVAL_FEEDBACK_BANDIT_EPSILON_MIN": "0.05",
            }.get(key, default)

        def set(self, key, value, actor="system", source="internal"):
            updates.append((key, value, actor, source))

    monkeypatch.setattr(ef, "_settings", lambda: _Settings())
    monkeypatch.setattr(ef, "_persist_feedback", lambda payload: None)
    monkeypatch.setattr(ef, "_theme_quality_breakdown", lambda run_id: {"historia": 4.5})
    monkeypatch.setattr(ef, "_apply_nsga_feedback", lambda *args, **kwargs: [])

    out = ef.apply_eval_run_feedback("eval_test", summary={"quality_mean": 4.5, "latency_mean": 2.0, "cost_mean": 0.02})
    assert out["run_id"] == "eval_test"
    assert "historia" in out["theme_breakdown"]
    assert any(item[0] == "BANDIT_EPSILON" for item in updates)


def test_get_latest_eval_feedback_without_redis(monkeypatch):
    from app.services import eval_feedback as ef

    monkeypatch.setattr(ef, "_redis_client", lambda: None)
    assert ef.get_latest_eval_feedback() is None
