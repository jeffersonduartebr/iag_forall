# Objective: Pure OpenRouter exploration policy (config parsing, gates, filters, UCB, running stats).
"""Behaviour of ``app.openrouter_exploration_policy`` — no I/O besides the in-memory catalog cache."""

from __future__ import annotations

import types

import pytest

from app import openrouter_catalog as cat
from app import openrouter_exploration_policy as pol


def _cfg(**overrides):
    base = {"OPENROUTER_EXPLORATION_ENABLED": "1", "OPENROUTER_EXPLORATION_PROMOTE_MIN_SAMPLES": "3"}
    base.update(overrides)
    return pol.load_exploration_config(base)


GOOD = {"count": 5, "mean_reward": 0.9, "mean_latency_s": 2.0, "mean_observed_usd_per_1k": 0.001,
        "failure_count": 0, "mean_judge_quality": 8.0}


@pytest.fixture
def priced(monkeypatch):
    monkeypatch.setitem(cat._CACHE, "data", [
        {"id": "cheap/m", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
        {"id": "pricey/m", "pricing": {"prompt": "0.001", "completion": "0.001"}},
        {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
    ])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [('["a", " b ", ""]', ("a", "b")), ("a, b,,", ("a", "b")), ("[broken", ("[broken",)),
     (["x", " "], ("x",)), (None, ()), ('{"a": 1}', ('{"a": 1}',))],
)
def test_parse_allowlist_accepts_json_csv_and_lists(raw, expected):
    assert pol._parse_allowlist(raw) == expected


def test_config_defaults_are_clamped_and_attribute_settings_work():
    ns = types.SimpleNamespace(OPENROUTER_EXPLORATION_RATE="5", OPENROUTER_EXPLORATION_MAX_PER_DAY="0",
                               OPENROUTER_EXPLORATION_POOL_SIZE="1", OPENROUTER_EXPLORATION_MODE="  ")
    cfg = pol.load_exploration_config(ns)
    assert cfg.enabled is False and cfg.rate == 1.0 and cfg.mode == "balanced"
    assert cfg.max_per_day == 1 and cfg.pool_size == 5 and cfg.pool_cache_ttl_s == 600
    assert cfg.provider_allowlist == () and cfg.auto_promote_enabled and cfg.promote_eval_gate_enabled


def test_config_survives_frozen_policy_lookup_failure(monkeypatch):
    def boom(_):
        raise RuntimeError("no frozen policy store")

    monkeypatch.setattr("app.services.frozen_policy.frozen_exploration_enabled", boom)
    cfg = _cfg(OPENROUTER_EXPLORATION_RATE="0.3", OPENROUTER_EXPLORATION_PROVIDER_ALLOWLIST="openai,anthropic")
    assert cfg.enabled is True and cfg.rate == pytest.approx(0.3)
    assert cfg.provider_allowlist == ("openai", "anthropic")


def test_frozen_policy_turns_exploration_off(monkeypatch):
    monkeypatch.setattr("app.services.frozen_policy.is_frozen_policy_active", lambda *a: True)
    cfg = _cfg(OPENROUTER_EXPLORATION_RATE="0.5")
    assert cfg.enabled is False and cfg.rate == 0.0


def test_promotion_passes_every_gate_for_a_good_model():
    promo = pol.evaluate_promotion(GOOD, _cfg())
    assert promo["promotable"] is True and promo["promotion_blockers"] == []
    assert "custo_ok" in promo["promotion_passed"] and "eval_gate_ok" in promo["promotion_passed"]


@pytest.mark.parametrize(
    ("patch", "blocker"),
    [({"count": 2}, "amostras_insuficientes"), ({"mean_reward": 0.1}, "reward_baixo"),
     ({"mean_latency_s": 99.0}, "latencia_alta"), ({"mean_observed_usd_per_1k": 0.5}, "custo_alto"),
     ({"failure_count": 2}, "taxa_falha_alta"), ({"mean_judge_quality": 3.0}, "eval_gate_judge_baixo"),
     ({"mean_judge_quality": None}, "eval_gate_sem_judge")],
)
def test_each_promotion_gate_blocks_on_its_own(patch, blocker):
    promo = pol.evaluate_promotion({**GOOD, **patch}, _cfg())
    assert promo["promotable"] is False and promo["promotion_blockers"] == [blocker]


def test_eval_gate_can_be_disabled_and_unknown_cost_is_neutral():
    stats = {**GOOD, "mean_judge_quality": None, "mean_observed_usd_per_1k": None}
    promo = pol.evaluate_promotion(stats, _cfg(OPENROUTER_EXPLORATION_PROMOTE_EVAL_GATE_ENABLED="0"))
    assert promo["promotable"] is True and "custo_ok" not in promo["promotion_passed"]
    assert pol._failure_rate({"count": 0, "failure_count": 3}) == 0.0


def test_cost_tier_bands():
    assert pol._cost_tier(0.5, 1.0) == "mais_barato"
    assert pol._cost_tier(1.1, 1.0) == "similar"
    assert pol._cost_tier(2.0, 1.0) == "mais_caro"
    assert pol._cost_tier(None, 1.0) == pol._cost_tier(1.0, 0.0) == "desconhecido"


def test_cost_comparison_uses_pool_median_of_observed_costs():
    models = [{"mean_observed_usd_per_1k": v, "observed_cost_samples": 1} for v in (0.001, 0.002, 0.004, 0.010)]
    bench = pol._enrich_models_with_cost_comparison(models)
    assert bench["pool_median_observed_usd_per_1k"] == pytest.approx(0.003)  # mediana par
    assert [m["cost_tier"] for m in models] == ["mais_barato", "mais_barato", "mais_caro", "mais_caro"]


def test_cost_comparison_falls_back_to_catalog_price_without_observations():
    models = [{"catalog_usd_per_1k": {"prompt_usd_per_1k": 0.001, "completion_usd_per_1k": 0.003},
               "mean_observed_usd_per_1k": 0.002}, {}]
    bench = pol._enrich_models_with_cost_comparison(models)
    assert bench["pool_median_observed_usd_per_1k"] is None
    assert models[0]["catalog_blended_usd_per_1k"] == 0.002
    assert models[0]["cost_vs_pool_ratio"] == 1.0 and models[0]["cost_tier"] == "similar"
    assert models[1]["cost_vs_pool_ratio"] is None and models[1]["cost_tier"] == "desconhecido"


def test_adaptive_rate_scales_with_uncertainty_and_is_clamped():
    cfg = _cfg(OPENROUTER_EXPLORATION_RATE="0.8")
    assert pol._effective_exploration_rate(cfg, 0.0) == pytest.approx(0.4)
    assert pol._effective_exploration_rate(cfg, 7.0) == 1.0
    fixed = _cfg(OPENROUTER_EXPLORATION_RATE="0.8", OPENROUTER_EXPLORATION_ADAPTIVE_RATE_ENABLED="0")
    assert pol._effective_exploration_rate(fixed, 1.0) == pytest.approx(0.8)


def test_price_cap_filters_expensive_and_variable_priced_models(priced):
    cfg = _cfg(OPENROUTER_EXPLORATION_MAX_PRICE_PROMPT_1K="0.01", OPENROUTER_EXPLORATION_MAX_PRICE_COMPLETION_1K="0.03")
    assert pol._price_within_budget("cheap/m", cfg) is True
    assert pol._price_within_budget("pricey/m", cfg) is False
    assert pol._price_within_budget("unknown/m", cfg) is True  # sem preço no catálogo: não bloqueia
    # Regressão: "-1" (preço variável, ex. openrouter/auto) virava -1.0/1k e passava por qualquer teto.
    assert pol._price_within_budget("openrouter/auto", cfg) is False


def test_catalog_pricing_lookup_strips_the_openrouter_prefix(priced):
    assert pol._catalog_usd_per_1k_for_model("openrouter/cheap/m") == pytest.approx(
        {"prompt_usd_per_1k": 0.001, "completion_usd_per_1k": 0.002})
    assert pol._catalog_usd_per_1k_for_model("openrouter/none/m") is None
    assert pol._model_slug("plain/model") == "plain/model"


def test_provider_allowlist_matches_the_slug_prefix():
    cfg = _cfg(OPENROUTER_EXPLORATION_PROVIDER_ALLOWLIST='["openai"]')
    assert pol._provider_allowed("openai/gpt-4o-mini", cfg) is True
    assert pol._provider_allowed("openai-evil/x", cfg) is False
    assert pol._provider_allowed("openai", cfg) is True
    assert pol._provider_allowed("anything/x", _cfg()) is True


def test_ucb_tries_unseen_models_first_then_balances_reward_and_uncertainty():
    assert pol._pick_ucb_from_pool(["a", "b"], {"a": {"count": 3, "mean_reward": 0.9}}) == "b"
    stats = {"a": {"count": 50, "mean_reward": 0.6}, "b": {"count": 2, "mean_reward": 0.5}}
    assert pol._pick_ucb_from_pool(["a", "b"], stats) == "b"  # bônus de exploração domina
    stats = {"a": {"count": 50, "mean_reward": 0.9}, "b": {"count": 50, "mean_reward": 0.2}}
    assert pol._pick_ucb_from_pool(["a", "b"], stats) == "a"


def _step(stats, *, judge=None, success=True, cost=0.001, tokens=(500, 500), catalog=None):
    return pol.next_exploration_stats(stats, reward=0.8 if success else 0.0, latency_s=1.0, cost_usd=cost,
                                      prompt_tokens=tokens[0], completion_tokens=tokens[1], success=success,
                                      judge_quality=judge, catalog_usd_per_1k=catalog)


def test_judge_mean_counts_only_judged_outcomes():
    """Regression: the judge mean divided by *all* outcomes and was erased by any unjudged one."""
    stats = _step({}, judge=9.0)
    for _ in range(8):
        stats = _step(stats)  # feedback heurístico, sem juiz
        assert stats["mean_judge_quality"] == 9.0 and stats["judge_samples"] == 1
    stats = _step(stats, judge=3.0)
    assert stats["judge_samples"] == 2 and stats["mean_judge_quality"] == pytest.approx(6.0)
    assert stats["last_judge_quality"] == 3.0
    gate = pol.evaluate_promotion({**GOOD, **stats, "count": 10}, _cfg())
    assert gate["promotion_blockers"] == ["eval_gate_judge_baixo"]  # 6.0 < 6.5; antes dava 8.4 e promovia


def test_legacy_judge_mean_without_sample_count_weighs_as_one_sample():
    stats = _step({"count": 4, "mean_judge_quality": 8.0}, judge=6.0)
    assert stats["judge_samples"] == 2 and stats["mean_judge_quality"] == 7.0


def test_auto_promotion_mark_survives_later_outcomes():
    """Regression: the next outcome dropped ``auto_promoted_at``, so the model could be promoted again."""
    stats = _step({"count": 20, "auto_promoted_at": 123.0})
    assert stats["auto_promoted_at"] == 123.0
    assert "auto_promoted_at" not in _step({})


def test_failures_count_up_and_reset_on_success():
    stats = _step(_step({}, success=False, cost=0.0, tokens=(0, 0)), success=False, cost=0.0, tokens=(0, 0))
    assert stats["consecutive_failures"] == 2 and stats["failure_count"] == 2 and stats["failure_rate"] == 1.0
    assert "mean_observed_usd_per_1k" not in stats  # nenhum token observado ainda
    stats = _step(stats, catalog={"prompt_usd_per_1k": 0.0011111111, "completion_usd_per_1k": 0.002})
    assert stats["consecutive_failures"] == 0 and stats["failure_count"] == 2
    assert stats["mean_observed_usd_per_1k"] == pytest.approx(0.001) and stats["observed_cost_samples"] == 1
    assert stats["catalog_usd_per_1k"]["prompt_usd_per_1k"] == 0.00111111
    stats = _step(stats, success=False, cost=0.0, tokens=(0, 0))  # falha sem tokens mantém o custo observado
    assert stats["mean_observed_usd_per_1k"] == pytest.approx(0.001) and stats["observed_cost_samples"] == 1
