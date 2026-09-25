# Objective: Adversarial governance — Redis-backed risk memory, degraded stores, online-loop wiring, escalation.
"""Coverage of app.services.adversarial_governance paths the base suite leaves untested."""

from __future__ import annotations

import json

import app.services.adversarial_governance as ag
import pytest


class _Settings:
    def __init__(self, values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


BASE = {
    "ADVGOV_ENABLED": "1", "ADVGOV_FAIL_SCORE_THRESHOLD": "7", "ADVGOV_CLUSTER_MIN_SAMPLES": "2",
    "ADVGOV_CLUSTER_FAILURE_RATE_THRESHOLD": "0.5", "ADVGOV_CLUSTER_TTL_S": "5", "UNCERTAINTY_THRESHOLD": "0.7",
    "ADVGOV_ESCALATION_MODELS": "[]",
}


@pytest.fixture
def gov(monkeypatch):
    values = dict(BASE)
    monkeypatch.setattr(ag, "_settings", lambda: _Settings(values))
    monkeypatch.setattr(ag, "_close_online_loop", lambda **kw: None)
    ag.reset_state()
    yield values
    ag.reset_state()


def test_cluster_risk_survives_process_restart_via_redis(gov, fake_redis):
    for _ in range(2):
        ag.record_adversarial_outcome(cluster_id="calculo", model="ollama/a", score=1.0, strategy="jailbreak")
    ag.reset_state()  # new worker: in-memory state gone, Redis remains
    risk = ag.get_cluster_risk("calculo")
    assert risk["high_risk"] is True and risk["n"] == 2
    assert risk["last_strategy"] == "jailbreak"
    assert 0 < fake_redis.ttl("advgov:cluster:calculo") <= 60  # TTL floor of 60 s over a 5 s setting


def test_corrupted_redis_value_falls_back_to_memory(gov, fake_redis):
    ag.record_adversarial_outcome(cluster_id="c", model="m", score=9.0)
    fake_redis.set("advgov:cluster:c", "{not json")
    assert ag.get_cluster_risk("c")["n"] == 1


def test_redis_write_failure_keeps_memory_copy(gov, monkeypatch):
    class _BrokenRedis:
        def get(self, _key):
            return None

        def set(self, *a, **k):
            raise ConnectionError("down")

    monkeypatch.setattr(ag, "_read_redis", lambda: _BrokenRedis())
    risk = ag.record_adversarial_outcome(cluster_id="c", model="m", score=2.0)
    assert risk["failures"] == 1 and ag._MEM_CLUSTERS["c"]["n"] == 1


def test_read_redis_swallows_client_errors(monkeypatch):
    def _boom():
        raise RuntimeError("no redis")

    monkeypatch.setattr("app.utils.redis_client.get_redis_sync_nonblocking", _boom)
    assert ag._read_redis() is None


def test_bad_score_is_reported_not_raised(gov, monkeypatch):
    monkeypatch.setattr(ag, "_read_redis", lambda: None)
    assert ag.record_adversarial_outcome(cluster_id="c", model="m", score="not-a-number") is None
    assert ag.get_cluster_risk(None)["high_risk"] is False


def test_settings_helpers_degrade_to_defaults(monkeypatch):
    def _boom():
        raise RuntimeError("settings down")

    monkeypatch.setattr(ag, "_settings", _boom)
    assert ag._cfg_bool("X", True) is True
    assert ag._cfg_float("X", 1.5) == 1.5
    assert ag._cfg_int("X", 3) == 3
    assert ag._cfg_list("X") == []
    assert ag._enabled() is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(["a", " ", "b"], ["a", "b"]), ("", []), ('["x","y"]', ["x", "y"]), ("x, y,,", ["x", "y"]), ('{"k":1}', ['{"k":1}'])],
)
def test_cfg_list_accepts_list_json_and_csv(monkeypatch, raw, expected):
    monkeypatch.setattr(ag, "_settings", lambda: _Settings({"L": raw}))
    assert ag._cfg_list("L") == expected


class _Predictor:
    def __init__(self, calls):
        self.calls = calls

    def predict_error_probability(self, emb):
        self.calls.append(("predict", list(emb)))
        return 0.3

    def learn(self, emb, ok):
        self.calls.append(("learn", ok))

    def record_outcome(self, pred, failed):
        self.calls.append(("outcome", pred, failed))

    def maybe_save(self):
        self.calls.append(("save",))


@pytest.fixture
def loop(monkeypatch):
    calls = []
    monkeypatch.setattr("app.bandits.compute_reward", lambda m, s, lat, cost: s / 10)
    monkeypatch.setattr("app.bandits.bandit_update", lambda **kw: calls.append(("bandit", kw["reward"])))
    monkeypatch.setattr("app.embeddings.embed_text", lambda q: calls.append(("embed", q)) or [0.1, 0.2])
    monkeypatch.setattr("app.online_predictor.get_predictor", lambda model: _Predictor(calls))
    return calls


def test_online_loop_predicts_before_learning_a_failure(loop):
    ag._close_online_loop(model="m", score=3.0, is_failure=True, query="q", embedding=None, latency_s=None,
                          cost_per_1k=None)
    assert loop == [("bandit", 0.3), ("embed", "q"), ("predict", [0.1, 0.2]), ("learn", False),
                    ("outcome", 0.3, True), ("save",)]


def test_online_loop_uses_given_embedding_and_survives_bandit_failure(loop, monkeypatch):
    def _boom(**kw):
        raise RuntimeError("bandit store down")

    monkeypatch.setattr("app.bandits.bandit_update", _boom)
    ag._close_online_loop(model="m", score=9.0, is_failure=False, query="q", embedding=(1.0,), latency_s=1.0,
                          cost_per_1k=0.1)
    assert ("embed", "q") not in loop
    assert ("learn", True) in loop


def test_online_loop_swallows_predictor_errors(loop, monkeypatch):
    def _boom(model):
        raise RuntimeError("river missing")

    monkeypatch.setattr("app.online_predictor.get_predictor", _boom)
    ag._close_online_loop(model="m", score=1.0, is_failure=True, query=None, embedding=[0.5], latency_s=None,
                          cost_per_1k=None)
    assert loop == [("bandit", 0.1)]


def _risky(gov, monkeypatch):
    monkeypatch.setattr(ag, "_read_redis", lambda: None)
    for _ in range(2):
        ag.record_adversarial_outcome(cluster_id="fis", model="m", score=0.0)


def test_escalation_from_cloud_model_moves_to_another_cloud_model(gov, monkeypatch):
    _risky(gov, monkeypatch)
    out = ag.suggest_escalation(cluster_id="fis", candidate_models=["openai/a", "ollama/b", "anthropic/c"],
                                current_model="openai/a")
    assert out == "anthropic/c"


def test_escalation_is_none_when_only_local_or_no_candidates(gov, monkeypatch):
    _risky(gov, monkeypatch)
    assert ag.suggest_escalation(cluster_id="fis", candidate_models=["ollama/a", "ollama/b"]) is None
    assert ag.suggest_escalation(cluster_id="fis", candidate_models=["", None]) is None


def test_configured_target_equal_to_current_is_skipped(gov, monkeypatch):
    _risky(gov, monkeypatch)
    gov["ADVGOV_ESCALATION_MODELS"] = json.dumps(["openai/strong", "anthropic/other"])
    out = ag.suggest_escalation(cluster_id="fis", candidate_models=["openai/strong", "anthropic/other", "ollama/x"],
                                current_model="openai/strong")
    assert out == "anthropic/other"


def test_router_hook_escalates_on_theme_and_tolerates_bad_logger(gov, monkeypatch):
    _risky(gov, monkeypatch)
    deps = {"logger": None}  # .info() raises; must not break routing
    chosen, top2 = ag.advgov_escalate(deps, "ollama/x", ["ollama/x", "ollama/y"], ["ollama/x", "openai/z"], None,
                                      {"benchmark_theme": "fis"})
    assert (chosen, top2) == ("openai/z", ["openai/z"])


def test_router_hook_keeps_choice_when_calm_or_on_error(gov, monkeypatch):
    monkeypatch.setattr(ag, "_read_redis", lambda: None)
    args = ({}, "ollama/x", ["ollama/x"], ["ollama/x", "openai/z"], 0.1)
    assert ag.advgov_escalate(*args, {"theme": "calm"}) == ("ollama/x", ["ollama/x"])

    def _boom(**kw):
        raise RuntimeError("store down")

    monkeypatch.setattr(ag, "suggest_escalation", _boom)
    assert ag.advgov_escalate(*args, None) == ("ollama/x", ["ollama/x"])
