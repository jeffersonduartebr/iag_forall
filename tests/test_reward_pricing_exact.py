# Objective: Exact-value tests for the reward transfer functions and the pricing/cost imputation.
"""services.reward and utils.pricing with pinned numbers (mutation-testing targets)."""

import json
import math
from contextlib import contextmanager
from types import SimpleNamespace

import fakeredis
import pytest
from app.observability import REWARD_WEIGHT, REWARD_WEIGHTS_SOURCE
from app.services import reward
from app.utils import pricing

# ---------------------------------------------------------------- recompensa


@pytest.mark.parametrize(
    ("shares", "floor", "expected"),
    [
        ((0.9, 0.08, 0.02), 0.1, (0.8, 0.1, 0.1)),
        ((0.85, 0.1, 0.05), 0.1, (0.8, 0.1, 0.1)),  # duas rodadas: o reescalonamento derruba a 2ª parcela
        ((0.5, 0.3, 0.2), 0.1, (0.5, 0.3, 0.2)),
        ((0.0, 0.0, 1.0), 0.2, (0.2, 0.2, 0.6)),
    ],
)
def test_apply_floor_exact(shares, floor, expected):
    assert reward._apply_floor(shares, floor) == pytest.approx(expected)


def test_normalize_and_parse_weights():
    assert reward._normalize((2, "1", 1)) == pytest.approx((0.5, 0.25, 0.25))
    assert reward._normalize((-1, 1, 1)) == pytest.approx((0.0, 0.5, 0.5))
    assert reward._normalize((math.inf, 1, 1)) == pytest.approx((0.0, 0.5, 0.5))
    assert reward._normalize((0, 0, 0)) is None
    assert reward._normalize((1, 1)) is None
    assert reward._normalize((1, "x", 1)) is None
    assert reward._parse_weights(b'{"quality": 3, "latency": 1, "cost": 0}') == pytest.approx((0.75, 0.25, 0.0))
    assert reward._parse_weights("[1, 2, 3]") is None
    assert reward._parse_weights("{quebrado") is None
    assert reward._parse_weights("") is None


def test_setting_float_guards(monkeypatch):
    values = {"A": "2.5", "B": "inf", "C": "abc"}
    monkeypatch.setattr(reward, "settings", SimpleNamespace(get=lambda key, default=None: values.get(key, default)))
    assert reward._setting_float("A", 1.0) == 2.5
    assert reward._setting_float("B", 1.0) == 1.0
    assert reward._setting_float("C", 1.0) == 1.0
    assert reward._setting_float("D", 7.0) == 7.0


def test_latency_and_cost_scores_exact():
    assert reward.latency_score(reward.LATENCY_X0_S) == pytest.approx(0.5)
    assert reward.latency_score(0.0) == pytest.approx(1 / (1 + math.exp(-reward.LATENCY_K * reward.LATENCY_X0_S)))
    assert reward.latency_score(1e5) == 0.0  # guarda de overflow
    assert reward.cost_score(None, 0.007) == 1.0
    assert reward.cost_score(0.007, 0.007) == 1.0
    assert reward.cost_score(0.014, 0.007) == pytest.approx(0.5)
    assert reward.cost_score(1.0, 0.007) == reward.COST_SCORE_FLOOR
    assert reward.cost_score(5.0, 0.0) == 1.0  # baseline inválido: neutro


def test_cost_helpers_exact():
    assert reward.cost_per_1k_from_total(0.002, 500, 500) == pytest.approx(0.002)
    assert reward.cost_per_1k_from_total(-1.0, 10, 0) == 0.0
    assert reward.cost_per_1k_from_total(math.inf, 10, 0) is None
    assert reward.cost_per_1k_from_total(0.1, "x", 0) is None
    assert reward.cost_per_1k_from_total(0.1, None, None) is None
    assert reward.blended_cost_per_1k(0.001, 0.004, 0.75) == pytest.approx(0.00175)
    assert reward.blended_cost_per_1k(0.001, 0.004, 1.5) == pytest.approx(0.001)
    assert reward.blended_cost_per_1k(0.001, 0.004, -1.0) == pytest.approx(0.004)
    prices = [(0.001, 0.004), (0.004, 0.016), (0.0, 0.0)]
    assert reward.calibrate_cost_baseline(prices, 0.75) == pytest.approx(math.sqrt(0.00175 * 0.007))
    assert reward.calibrate_cost_baseline([], 0.75) == reward.DEFAULT_COST_BASELINE_PER_1K


def test_compute_reward_clamps_quality_and_survives_errors(monkeypatch):
    monkeypatch.setattr(reward, "_setting_float", lambda key, default: default)
    monkeypatch.setattr(reward, "load_reward_weights", lambda modality="text": ((0.5, 0.5, 0.0), "test"))
    assert reward.compute_reward("m", 0.5, 1e5) == pytest.approx(0.025)  # latência saturada: só qualidade
    assert reward.compute_reward("m", 10.5, 1e5) == pytest.approx(0.5)  # nota > 10 limitada
    assert reward.compute_reward("m", -3.0, 1e5) == 0.0

    def broken(modality="text"):
        raise RuntimeError("redis down")

    monkeypatch.setattr(reward, "load_reward_weights", broken)
    assert reward.compute_reward("m", 8.0, 1.0) == 0.0


def _source_count(source):
    return REWARD_WEIGHTS_SOURCE.labels(source=source)._value.get()


def test_load_reward_weights_lookup_order_and_metrics(monkeypatch):
    rds = fakeredis.FakeRedis()
    monkeypatch.setattr(reward, "_get_rds", lambda: rds)
    rds.set(reward.REWARD_WEIGHTS_KEY.format(modality="text"), json.dumps({"quality": 1, "latency": 1, "cost": 2}))

    before = _source_count("nsga")
    weights, source = reward.load_reward_weights(" VISION ")  # sem pesos de vision: usa os de text
    assert (weights, source) == ((0.25, 0.25, 0.5), "nsga")
    assert _source_count("nsga") == before + 1
    assert REWARD_WEIGHT.labels(objective="cost", modality="vision")._value.get() == 0.5

    rds.set(reward.REWARD_WEIGHTS_KEY.format(modality="vision"), json.dumps({"quality": 1, "latency": 0, "cost": 0}))
    assert reward.load_reward_weights("vision") == ((1.0, 0.0, 0.0), "nsga")
    assert reward.load_reward_weights("") == ((0.25, 0.25, 0.5), "nsga")  # vazio vira text


def test_load_reward_weights_stops_on_redis_error(monkeypatch):
    calls = []

    class _Broken:
        def get(self, key):
            calls.append(key)
            raise ConnectionError("down")

    monkeypatch.setattr(reward, "_get_rds", lambda: _Broken())
    before = _source_count("default")
    assert reward.load_reward_weights("vision") == (reward.DEFAULT_REWARD_WEIGHTS, "default")
    assert calls == [reward.REWARD_WEIGHTS_KEY.format(modality="vision")]  # não insiste com text
    assert _source_count("default") == before + 1
    assert REWARD_WEIGHT.labels(objective="quality", modality="vision")._value.get() == 0.55


def test_fallback_warning_is_rate_limited(monkeypatch, fake_clock, caplog):
    clock = fake_clock(reward)
    monkeypatch.setattr(reward, "_last_fallback_warn", 0.0)
    monkeypatch.setattr(reward, "_get_rds", lambda: None)
    with caplog.at_level("WARNING", logger=reward.logger.name):
        reward.load_reward_weights("text")
        clock.advance(reward._FALLBACK_WARN_INTERVAL_S - 1)
        reward.load_reward_weights("text")
        clock.advance(1)
        reward.load_reward_weights("text")
    warnings = [r for r in caplog.records if "Sem pesos publicados" in r.getMessage()]
    assert len(warnings) == 2
    assert "nsga:reward_weights:text" in warnings[0].getMessage()


def test_publish_reward_weights_payload(monkeypatch, fake_clock):
    fake_clock(reward)
    rds = fakeredis.FakeRedis()
    assert reward.publish_reward_weights(rds, "text", (0.5, 0.3333333333, 0.1666666667), {"lat": 2.0}) is True
    payload = json.loads(rds.get(reward.REWARD_WEIGHTS_KEY.format(modality="text")))
    assert payload == {
        "quality": 0.5,
        "latency": 0.333333,
        "cost": 0.166667,
        "updated_at": 1_000_000.0,
        "source": "nsga-updater",
        "sys_metrics": {"lat": 2.0},
    }
    assert reward.publish_reward_weights(None, "text", (1, 0, 0)) is False
    broken = SimpleNamespace(set=lambda *a: (_ for _ in ()).throw(ConnectionError()))
    assert reward.publish_reward_weights(broken, "text", (1, 0, 0)) is False


# ---------------------------------------------------------------- preços e custo local


def test_local_cost_rate_formula(monkeypatch):
    values = {
        "LOCAL_COST_HW_PRICE_USD": 3000.0,
        "LOCAL_COST_HW_LIFETIME_YEARS": 3.0,
        "LOCAL_COST_HW_UTILIZATION": 0.5,
        "LOCAL_COST_POWER_W": 300.0,
        "LOCAL_COST_ENERGY_USD_PER_KWH": 0.2,
    }
    monkeypatch.setattr(pricing, "_local_cost_setting", lambda key, default: values.get(key, default))
    assert pricing.local_cost_rate_usd_per_hour() == pytest.approx(3000 / (3 * 8760 * 0.5) + 0.3 * 0.2)

    values["LOCAL_COST_HW_UTILIZATION"] = 0.0  # sem horas produtivas: só energia
    assert pricing.local_cost_rate_usd_per_hour() == pytest.approx(0.06)
    values.update(LOCAL_COST_HW_PRICE_USD=-10.0, LOCAL_COST_POWER_W=-5.0, LOCAL_COST_HW_UTILIZATION=0.5)
    assert pricing.local_cost_rate_usd_per_hour() == 0.0
    values["LOCAL_COST_USD_PER_HOUR"] = 0.9  # valor explícito tem precedência
    assert pricing.local_cost_rate_usd_per_hour() == 0.9


def test_impute_local_cost_guards(monkeypatch):
    values = {"LOCAL_COST_USD_PER_HOUR": 0.36, "LOCAL_COST_PARALLEL_SLOTS": 2.0}
    monkeypatch.setattr(pricing, "_local_cost_setting", lambda key, default: values.get(key, default))
    assert pricing.impute_local_cost(3600) == pytest.approx(0.18)
    assert pricing.impute_local_cost("x") == 0.0
    assert pricing.impute_local_cost(math.nan) == 0.0
    assert pricing.impute_local_cost(-5) == 0.0
    values["LOCAL_COST_PARALLEL_SLOTS"] = 0.5  # menos de 1 slot conta como 1
    assert pricing.impute_local_cost(3600) == pytest.approx(0.36)
    values["LOCAL_COST_IMPUTATION_ENABLED"] = 0.0
    assert pricing.impute_local_cost(3600) == 0.0


def test_local_cost_setting_guards(monkeypatch):
    values = {"A": "1.5", "B": "nan", "C": "x"}
    monkeypatch.setattr("app.settings_dynamic.settings", SimpleNamespace(get=lambda k, d=None: values.get(k, d)))
    assert pricing._local_cost_setting("A", 9.0) == 1.5
    assert pricing._local_cost_setting("B", 9.0) == 9.0
    assert pricing._local_cost_setting("C", 9.0) == 9.0
    assert pricing.is_local_model("Ollama/gemma") and not pricing.is_local_model(None)


@pytest.fixture
def pricing_env(monkeypatch):
    rds = fakeredis.FakeRedis()
    monkeypatch.setattr(pricing, "_get_rds", lambda: rds)
    monkeypatch.setattr(pricing, "_PRICING_CACHE", {})
    monkeypatch.setattr(pricing, "_LAST_UPDATE", 0.0)
    return rds


def _engine(rows):
    conn = SimpleNamespace(execute=lambda stmt: SimpleNamespace(fetchall=lambda: rows))

    @contextmanager
    def connect():
        yield conn

    return SimpleNamespace(connect=connect)


def test_refresh_pricing_prefers_redis_then_db(monkeypatch, pricing_env):
    rds = pricing_env
    monkeypatch.setattr(pricing, "get_engine", lambda: _engine([("x/m1", "0.001", "0.002")]))
    pricing._refresh_pricing()
    assert pricing._PRICING_CACHE == {"x/m1": {"in": 0.001, "out": 0.002}}
    assert json.loads(rds.get(pricing.REDIS_PRICING_KEY)) == pricing._PRICING_CACHE
    assert 0 < rds.ttl(pricing.REDIS_PRICING_KEY) <= pricing.REDIS_PRICING_TTL

    rds.set(pricing.REDIS_PRICING_KEY, json.dumps({"x/m2": {"in": 1.0, "out": 1.0}}))
    monkeypatch.setattr(pricing, "get_engine", lambda: pytest.fail("não deveria consultar o DB"))
    pricing._refresh_pricing()
    assert set(pricing._PRICING_CACHE) == {"x/m2"}


def test_refresh_pricing_db_failure_keeps_cache(monkeypatch, pricing_env):
    monkeypatch.setattr(pricing, "_PRICING_CACHE", {"keep": {"in": 1.0, "out": 1.0}})
    monkeypatch.setattr(pricing, "get_engine", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    pricing._refresh_pricing()
    assert pricing._PRICING_CACHE == {"keep": {"in": 1.0, "out": 1.0}}
    assert pricing._refresh_pricing_from_db() == {}


def test_invalidate_pricing_cache(pricing_env, monkeypatch):
    rds = pricing_env
    rds.set(pricing.REDIS_PRICING_KEY, "{}")
    monkeypatch.setattr(pricing, "_PRICING_CACHE", {"m": {"in": 1, "out": 1}})
    monkeypatch.setattr(pricing, "_LAST_UPDATE", 123.0)
    pricing.invalidate_pricing_cache()
    assert (pricing._PRICING_CACHE, pricing._LAST_UPDATE) == ({}, 0)
    assert rds.get(pricing.REDIS_PRICING_KEY) is None


def test_lookup_catalog_sources(monkeypatch, pricing_env):
    monkeypatch.setattr(pricing, "_PRICING_CACHE", {"gpt-x": {"in": 1.0, "out": 2.0}})
    assert pricing._lookup_catalog("openai/gpt-x") == {"in": 1.0, "out": 2.0}  # sem prefixo do provider
    monkeypatch.setattr(
        "app.openrouter_catalog.get_openrouter_pricing_per_1k",
        lambda slug: {"in": 0.5, "out": 0.5} if slug == "a/b" else None,
    )
    assert pricing._lookup_catalog("openrouter/a/b") == {"in": 0.5, "out": 0.5}
    assert pricing._lookup_catalog("openai/desconhecido") is None
    monkeypatch.setattr("app.openrouter_catalog.get_openrouter_pricing_per_1k", lambda slug: 1 / 0)
    assert pricing._lookup_catalog("openrouter/a/b") is None


def test_get_model_cost_refreshes_when_stale(monkeypatch, pricing_env):
    refreshed = []

    def refresh():
        refreshed.append(1)
        pricing._PRICING_CACHE["m"] = {"in": 0.002, "out": 0.004}

    monkeypatch.setattr(pricing, "_refresh_pricing", refresh)
    assert pricing.get_model_cost("m", 1000, 500) == pytest.approx(0.004)
    assert refreshed == [1]
    monkeypatch.setattr(pricing, "_LAST_UPDATE", pricing.time.time())
    assert pricing.get_model_cost("ollama/x", 1000, 1000) == 0.0
    assert refreshed == [1]
