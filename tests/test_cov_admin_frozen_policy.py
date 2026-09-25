# Objective: Behavioural coverage for frozen-policy mode (eval runs without exploration or tuning).
"""frozen_policy: snapshot, Redis lifecycle, exploration overrides and outage tolerance."""

import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def fp():
    from app.services import frozen_policy

    return frozen_policy


def test_snapshot_reads_routing_settings(fp, monkeypatch):
    values = {"BANDIT_EPSILON": None, "UNCERTAINTY_THRESHOLD": "0.4", "OPENROUTER_EXPLORATION_ENABLED": 1}
    fake = SimpleNamespace(NSGA_W_QUALITY="0.6", NSGA_W_LATENCY=0.3, NSGA_W_COST=0.1, get=values.get)
    monkeypatch.setattr("app.settings_dynamic.settings", fake)
    assert fp.build_frozen_snapshot() == {
        "NSGA_W_QUALITY": 0.6,
        "NSGA_W_LATENCY": 0.3,
        "NSGA_W_COST": 0.1,
        "BANDIT_EPSILON": 0.0,
        "UNCERTAINTY_THRESHOLD": 0.4,
        "OPENROUTER_EXPLORATION_ENABLED": True,
        "OPENROUTER_EXPLORATION_RATE": 0.0,
    }


def test_lifecycle_in_redis(fp, fake_redis):
    assert fp.is_frozen_policy_active() is False
    payload = fp.activate_frozen_policy("r1", snapshot={"BANDIT_EPSILON": 0.2})
    assert payload == {"BANDIT_EPSILON": 0.2, "run_id": "r1", "active": True}
    assert fake_redis.get("eval:frozen:active") == b"r1" and fake_redis.ttl("eval:frozen:r1") > 0

    assert fp.get_frozen_policy() == payload
    assert fp.get_frozen_policy("r1") == payload and fp.get_frozen_policy("r2") is None
    assert fp.frozen_bandit_epsilon(0.3) == 0.0
    assert fp.frozen_exploration_enabled(True) is False
    assert fp.frozen_exploration_rate(0.25) == 0.0

    fp.deactivate_frozen_policy("r1")
    assert fake_redis.get("eval:frozen:active") is None and fake_redis.get("eval:frozen:r1") is None
    assert fp.frozen_bandit_epsilon(0.3) == 0.3
    assert fp.frozen_exploration_enabled(True) is True
    assert fp.frozen_exploration_rate(0.25) == 0.25


def test_deactivating_other_run_keeps_active_marker(fp, fake_redis):
    fp.activate_frozen_policy("a", snapshot={"x": 1})
    fp.activate_frozen_policy("b", snapshot={"x": 1})
    fp.deactivate_frozen_policy("a")
    assert fake_redis.get("eval:frozen:active") == b"b"
    assert fp.is_frozen_policy_active() is True


def test_activate_without_snapshot_builds_one(fp, fake_redis, monkeypatch):
    monkeypatch.setattr(fp, "build_frozen_snapshot", lambda: {"NSGA_W_COST": 0.1})
    assert fp.activate_frozen_policy("r")["NSGA_W_COST"] == 0.1


def test_context_manager_clears_even_on_error(fp, fake_redis):
    with pytest.raises(RuntimeError):
        with fp.frozen_policy_context("r1", snapshot={"x": 1}) as payload:
            assert payload["run_id"] == "r1" and fp.is_frozen_policy_active("r1")
            raise RuntimeError("falhou a meio")
    assert fp.is_frozen_policy_active() is False


def test_corrupt_or_non_dict_payloads_read_as_inactive(fp, fake_redis):
    fake_redis.set("eval:frozen:active", "r1")
    fake_redis.set("eval:frozen:r1", json.dumps(["lista"]))
    assert fp.get_frozen_policy() is None
    fake_redis.set("eval:frozen:r1", "{quebrado")
    assert fp.is_frozen_policy_active() is False


def test_decoded_string_client(fp, monkeypatch):
    store = {"eval:frozen:active": "r1", "eval:frozen:r1": '{"run_id": "r1"}'}
    monkeypatch.setattr(fp, "_redis_client", lambda: SimpleNamespace(get=store.get, delete=store.pop))
    assert fp.get_frozen_policy() == {"run_id": "r1"}
    fp.deactivate_frozen_policy("r1")
    assert store == {}


def test_redis_unavailable_means_not_frozen(fp, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr("app.utils.redis_client.get_redis", boom)
    assert fp._redis_client() is None
    assert fp.get_frozen_policy() is None
    assert fp.activate_frozen_policy("r", snapshot={"x": 1}) == {"x": 1, "run_id": "r", "active": True}
    fp.deactivate_frozen_policy("r")

    broken = SimpleNamespace(set=boom, get=boom, delete=boom)
    monkeypatch.setattr(fp, "_redis_client", lambda: broken)
    assert fp.activate_frozen_policy("r", snapshot={"x": 1})["active"] is True
    fp.deactivate_frozen_policy("r")
    assert fp.get_frozen_policy("r") is None
    assert fp.frozen_bandit_epsilon(0.1) == 0.1


@pytest.mark.parametrize(
    ("meta", "skip"),
    [
        (None, False),
        ({}, False),
        ({"frozen_policy": True}, True),
        ({"experiment_manifest": {"frozen_policy": True}}, True),
        ({"experiment_manifest": None}, False),
    ],
)
def test_should_skip_eval_feedback(fp, meta, skip):
    assert fp.should_skip_eval_feedback(meta) is skip
