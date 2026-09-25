# Objective: Behavioural coverage for the eval-run -> routing-policy feedback bridge.
"""eval_feedback: frozen-run skip, epsilon clamping, NSGA diffs, theme means and Redis persistence."""

import json
import sys
from types import SimpleNamespace

import pytest


class _Settings:
    def __init__(self, **values):
        self.NSGA_W_QUALITY, self.NSGA_W_LATENCY, self.NSGA_W_COST = 1.0, 0.5, 0.2
        self.values = {"BANDIT_EPSILON": "0.10", **values}
        self.writes = []

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value, actor="system", source="internal"):
        self.values[key] = value
        self.writes.append((key, value, actor, source))


@pytest.fixture
def ef(monkeypatch):
    from app.services import eval_feedback

    monkeypatch.setattr(eval_feedback, "_theme_quality_breakdown", lambda run_id: {})
    monkeypatch.setattr(eval_feedback, "_apply_nsga_feedback", lambda *a: [])
    return eval_feedback


def _run(ef, monkeypatch, quality, settings=None, persisted=None):
    settings = settings or _Settings()
    monkeypatch.setattr(ef, "_settings", lambda: settings)
    monkeypatch.setattr(
        ef, "_persist_feedback", lambda payload: (persisted if persisted is not None else []).append(payload)
    )
    return ef.apply_eval_run_feedback("r1", summary={"quality_mean": quality}), settings


@pytest.mark.parametrize("meta", [{"frozen_policy": True}, {"experiment_manifest": {"frozen_policy": 1}}])
def test_frozen_runs_do_not_tune(ef, monkeypatch, meta):
    monkeypatch.setattr(ef, "_settings", lambda: pytest.fail("não devia ler settings"))
    assert ef.apply_eval_run_feedback("r1", summary={"quality_mean": 1}, metadata=meta) == {
        "run_id": "r1",
        "skipped": True,
        "reason": "frozen_policy",
    }


def test_broken_frozen_check_does_not_block_tuning(ef, monkeypatch):
    def boom(meta):
        raise RuntimeError("x")

    monkeypatch.setattr("app.services.frozen_policy.should_skip_eval_feedback", boom)
    out, _ = _run(ef, monkeypatch, 7.0)
    assert "skipped" not in out and out["bandit_changes"] == []


def test_high_quality_reduces_exploration_and_persists(ef, monkeypatch):
    persisted = []
    out, settings = _run(ef, monkeypatch, 9.0, persisted=persisted)
    assert settings.writes == [("BANDIT_EPSILON", "0.09", "eval-feedback", "eval_run")]
    assert out["bandit_changes"] == [{"setting": "BANDIT_EPSILON", "before": 0.1, "after": pytest.approx(0.09)}]
    assert persisted == [out]


@pytest.mark.parametrize(("quality", "eps"), [(9.0, "0.05"), (2.0, "0.35")])
def test_epsilon_already_at_bound_is_left_alone(ef, monkeypatch, quality, eps):
    out, settings = _run(ef, monkeypatch, quality, settings=_Settings(BANDIT_EPSILON=eps))
    assert settings.writes == [] and out["bandit_changes"] == []


def test_low_quality_raises_exploration_up_to_cap(ef, monkeypatch):
    out, settings = _run(ef, monkeypatch, 2.0, settings=_Settings(BANDIT_EPSILON="0.34"))
    assert settings.values["BANDIT_EPSILON"] == "0.35"


def test_quality_inside_band_changes_nothing(ef, monkeypatch):
    out, settings = _run(ef, monkeypatch, 7.0)
    assert settings.writes == [] and out["quality_mean"] == 7.0 and out["latency_mean"] == 0.0


def test_nsga_failure_is_reported_not_raised(ef, monkeypatch):
    def boom(*a):
        raise ValueError("deap indisponível")

    monkeypatch.setattr(ef, "_apply_nsga_feedback", boom)
    out, _ = _run(ef, monkeypatch, 7.0)
    assert out["nsga_error"] == "deap indisponível" and out["nsga_changes"] == []


def test_apply_nsga_feedback_reports_only_changed_weights(monkeypatch):
    from app.services.eval_feedback import _apply_nsga_feedback

    settings = _Settings()
    seen = []

    def tune(objectives):
        seen.append(objectives)
        settings.NSGA_W_QUALITY = 2.0

    monkeypatch.setitem(sys.modules, "app.nsga_weights_updater", SimpleNamespace(tune_global_strategy_weights=tune))
    changes = _apply_nsga_feedback(settings, 1.5, 0.01, 6.0)
    assert seen == [(1.5, 0.01, 6.0)]
    assert changes == [{"setting": "NSGA_W_QUALITY", "before": 1.0, "after": 2.0}]


def test_theme_breakdown_averages_by_theme(monkeypatch):
    from app.services.eval_feedback import _theme_quality_breakdown

    rows = [
        {"quality": 8, "metadata": {"benchmark_theme": "hist"}},
        {"quality": 6, "metadata": {"benchmark_theme": "hist"}},
        {"quality": None, "metadata": None},
    ]
    monkeypatch.setattr("app.roadmap_features.list_eval_run_results", lambda run_id, limit: rows)
    assert _theme_quality_breakdown("r1") == {"hist": 7.0, "unknown": 0.0}


def test_feedback_roundtrip_through_redis(fake_redis):
    from app.services import eval_feedback

    eval_feedback._persist_feedback({"run_id": "r9", "quality_mean": 5.0, "nota": "ação"})
    assert eval_feedback.get_latest_eval_feedback() == {"run_id": "r9", "quality_mean": 5.0, "nota": "ação"}
    assert 0 < fake_redis.ttl("eval:feedback:r9") <= 86400 * 14


def test_latest_feedback_edge_cases(fake_redis):
    from app.services import eval_feedback

    assert eval_feedback.get_latest_eval_feedback() is None
    fake_redis.set(eval_feedback.REDIS_EVAL_FEEDBACK_KEY, json.dumps([1, 2]))
    assert eval_feedback.get_latest_eval_feedback() is None
    fake_redis.set(eval_feedback.REDIS_EVAL_FEEDBACK_KEY, "{nao-json")
    assert eval_feedback.get_latest_eval_feedback() is None


def test_latest_feedback_accepts_str_from_decoding_client(monkeypatch):
    from app.services import eval_feedback

    monkeypatch.setattr(eval_feedback, "_redis_client", lambda: SimpleNamespace(get=lambda k: '{"run_id": "s"}'))
    assert eval_feedback.get_latest_eval_feedback() == {"run_id": "s"}


def test_redis_outage_is_swallowed(monkeypatch):
    from app.services import eval_feedback

    def boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr("app.utils.redis_client.get_redis", boom)
    assert eval_feedback._redis_client() is None
    assert eval_feedback.get_latest_eval_feedback() is None
    eval_feedback._persist_feedback({"run_id": "x"})  # sem cliente: não faz nada

    monkeypatch.setattr(eval_feedback, "_redis_client", lambda: SimpleNamespace(set=boom, get=boom))
    eval_feedback._persist_feedback({"run_id": "x"})
    assert eval_feedback.get_latest_eval_feedback() is None


def test_settings_accessor_returns_dynamic_settings():
    from app.services import eval_feedback
    from app.settings_dynamic import settings

    assert eval_feedback._settings() is settings
