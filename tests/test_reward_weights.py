# Objective: Test coverage for the NSGA-II-coupled bandit reward.
"""Reward weights published by the NSGA-II cycle and the reward transfer functions."""

import json
from types import SimpleNamespace

import pytest

from app.services import reward


class _FakeRedis:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value


def _weights_json(q, lat, c):
    return json.dumps({"quality": q, "latency": lat, "cost": c})


def test_derive_reward_weights_uses_contribution_shares():
    # W_Q*q = 1*8 = 8, W_L*l = 0.5*4 = 2, W_C*c = 100*0.02 = 2 -> 0.667 / 0.167 / 0.167
    wq, wl, wc = reward.derive_reward_weights(1.0, 0.5, 100.0, 8.0, 4.0, 0.02, min_share=0.0)
    assert (wq, wl, wc) == pytest.approx((8 / 12, 2 / 12, 2 / 12))


def test_derive_reward_weights_floors_small_shares_and_keeps_sum_one():
    # Custo ~0 (tudo local, sem imputação) não pode sumir da recompensa.
    weights = reward.derive_reward_weights(1.0, 0.5, 100.0, 8.0, 1.0, 0.0, min_share=0.05)
    assert sum(weights) == pytest.approx(1.0)
    assert min(weights) == pytest.approx(0.05)


def test_derive_reward_weights_degenerate_falls_back_to_default():
    assert reward.derive_reward_weights(0, 0, 0, 0, 0, 0) == reward.DEFAULT_REWARD_WEIGHTS


def test_load_reward_weights_prefers_modality_then_text_then_default(monkeypatch):
    rds = _FakeRedis(
        {
            "nsga:reward_weights:vision": _weights_json(2, 1, 1),
            "nsga:reward_weights:text": _weights_json(1, 1, 2),
        }
    )
    monkeypatch.setattr(reward, "_get_rds", lambda: rds)

    weights, source = reward.load_reward_weights("vision")
    assert source == "nsga" and weights == pytest.approx((0.5, 0.25, 0.25))

    weights, source = reward.load_reward_weights("multimodal")  # sem chave própria -> text
    assert source == "nsga" and weights == pytest.approx((0.25, 0.25, 0.5))

    monkeypatch.setattr(reward, "_get_rds", lambda: _FakeRedis())
    weights, source = reward.load_reward_weights("text")
    assert source == "default" and weights == reward.DEFAULT_REWARD_WEIGHTS


def test_legacy_per_model_key_is_not_mistaken_for_objective_weights(monkeypatch):
    # nsga:weights:<mod> guarda pesos POR MODELO; nunca deve alimentar a recompensa.
    rds = _FakeRedis({"nsga:weights:text": json.dumps({"ollama/phi4": 0.7, "openai/gpt-5.5": 0.3})})
    monkeypatch.setattr(reward, "_get_rds", lambda: rds)
    _, source = reward.load_reward_weights("text")
    assert source == "default"


def test_publish_then_load_roundtrip(monkeypatch):
    rds = _FakeRedis()
    assert reward.publish_reward_weights(rds, "text", (0.6, 0.25, 0.15), {"quality": 8.0})
    payload = json.loads(rds.data["nsga:reward_weights:text"])
    assert payload["source"] == "nsga-updater" and payload["sys_metrics"] == {"quality": 8.0}

    monkeypatch.setattr(reward, "_get_rds", lambda: rds)
    weights, source = reward.load_reward_weights("text")
    assert source == "nsga" and weights == pytest.approx((0.6, 0.25, 0.15))
    assert reward.publish_reward_weights(None, "text", (1, 0, 0)) is False


def test_compute_reward_uses_modality_weights(monkeypatch):
    seen = {}

    def _load(modality="text"):
        seen["modality"] = modality
        return (1.0, 0.0, 0.0), "nsga"

    monkeypatch.setattr(reward, "load_reward_weights", _load)
    assert reward.compute_reward("m", quality=7.0, latency_s=5.0, cost_per_1k=None, modality="vision") == pytest.approx(
        0.7
    )
    assert seen["modality"] == "vision"


def test_latency_score_is_logistic_and_overflow_safe():
    assert reward.latency_score(reward.LATENCY_X0_S) == pytest.approx(0.5)
    assert reward.latency_score(0.0) > 0.9
    assert reward.latency_score(1e6) == 0.0


def test_cost_per_1k_from_total():
    assert reward.cost_per_1k_from_total(0.001, 400, 100) == pytest.approx(0.002)
    assert reward.cost_per_1k_from_total(0.001, 0, 0) is None
    assert reward.cost_per_1k_from_total(None, 10, 10) == 0.0


def test_cost_baseline_reproduces_calibration():
    # Pool de CANDIDATE_MODELS_LIST a preços de lista (in/out por 1k), mix 3:1.
    pool = [(0.005, 0.030), (0.005, 0.025), (0.002, 0.010), (0.001, 0.005), (0.010, 0.050)]
    assert reward.calibrate_cost_baseline(pool, 0.75) == pytest.approx(reward.DEFAULT_COST_BASELINE_PER_1K, rel=0.02)
    assert reward.calibrate_cost_baseline([], 0.75) == reward.DEFAULT_COST_BASELINE_PER_1K


def test_cost_score_discriminates_current_pool():
    base = reward.DEFAULT_COST_BASELINE_PER_1K
    haiku = reward.cost_score(reward.blended_cost_per_1k(0.001, 0.005, 0.75), base)
    opus = reward.cost_score(reward.blended_cost_per_1k(0.005, 0.025, 0.75), base)
    fable = reward.cost_score(reward.blended_cost_per_1k(0.010, 0.050, 0.75), base)
    assert haiku == 1.0 > opus > fable >= reward.COST_SCORE_FLOOR
    # O baseline antigo (0.12) dava 1.0 para todos: termo de custo inerte.
    assert reward.cost_score(reward.blended_cost_per_1k(0.010, 0.050, 0.75), 0.12) == 1.0
    assert reward.cost_score(None, base) == 1.0


def test_compute_reward_reads_cost_baseline_setting(monkeypatch):
    monkeypatch.setattr(reward, "load_reward_weights", lambda modality="text": ((0.0, 0.0, 1.0), "nsga"))
    monkeypatch.setattr(reward, "settings", SimpleNamespace(get=lambda k, d=None: "0.01"))
    assert reward.compute_reward("m", 10.0, 1.0, cost_per_1k=0.02) == pytest.approx(0.5)


def test_publish_reward_shares_skips_under_frozen_policy(monkeypatch):
    from app import nsga_weights_updater as upd

    rds = _FakeRedis()
    monkeypatch.setattr(upd, "redis_client", rds)
    monkeypatch.setattr(upd, "is_frozen_policy_active", lambda: True)
    upd.publish_reward_shares("text", (2.0, 0.01, 8.0))
    assert rds.data == {}

    monkeypatch.setattr(upd, "is_frozen_policy_active", lambda: False)
    upd.publish_reward_shares("text", (2.0, 0.01, 8.0))
    payload = json.loads(rds.data["nsga:reward_weights:text"])
    assert payload["quality"] + payload["latency"] + payload["cost"] == pytest.approx(1.0)
