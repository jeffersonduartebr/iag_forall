# Objective: Test coverage for local inference cost imputation.
"""Imputed cost of local (Ollama) inference and its separation from cash cost."""

import pytest

from app.services.router_services import parse_meta_cost
from app.utils import pricing


def _settings(monkeypatch, values):
    monkeypatch.setattr(pricing, "_local_cost_setting", lambda key, default: float(values.get(key, default)))


def test_rate_from_components(monkeypatch):
    # 2628 USD / (3 anos * 8760 h * 0.5) = 0.2 USD/h ; 0.4 kW * 0.25 USD/kWh = 0.1 USD/h
    _settings(
        monkeypatch,
        {
            "LOCAL_COST_HW_PRICE_USD": 2628,
            "LOCAL_COST_HW_LIFETIME_YEARS": 3,
            "LOCAL_COST_HW_UTILIZATION": 0.5,
            "LOCAL_COST_POWER_W": 400,
            "LOCAL_COST_ENERGY_USD_PER_KWH": 0.25,
        },
    )
    assert pricing.local_cost_rate_usd_per_hour() == pytest.approx(0.3)
    assert pricing.impute_local_cost(36.0) == pytest.approx(0.003)


def test_hourly_override_and_parallel_slots(monkeypatch):
    _settings(monkeypatch, {"LOCAL_COST_USD_PER_HOUR": 0.36, "LOCAL_COST_PARALLEL_SLOTS": 2})
    assert pricing.local_cost_rate_usd_per_hour() == pytest.approx(0.36)
    assert pricing.impute_local_cost(10.0) == pytest.approx(0.0005)


def test_imputation_disabled_or_invalid_occupancy(monkeypatch):
    _settings(monkeypatch, {"LOCAL_COST_USD_PER_HOUR": 1.0, "LOCAL_COST_IMPUTATION_ENABLED": 0})
    assert pricing.impute_local_cost(100.0) == 0.0
    _settings(monkeypatch, {"LOCAL_COST_USD_PER_HOUR": 1.0})
    assert pricing.impute_local_cost(0) == 0.0
    assert pricing.impute_local_cost(float("nan")) == 0.0
    assert pricing.impute_local_cost("x") == 0.0


def test_is_local_model():
    assert pricing.is_local_model("ollama/gemma3:4b")
    assert not pricing.is_local_model("anthropic/claude-haiku-4-5-20251001")


def test_parse_meta_cost_adds_imputed_and_keeps_cash_separate():
    meta = {"prompt_tokens": 100, "completion_tokens": 50, "cost_per_1k": 0.0, "cost_imputed_usd": 0.002}
    p_tok, c_tok, total, _load, meta_safe = parse_meta_cost(meta, "ollama/gemma3:4b", lambda *a: 0.0)
    assert (p_tok, c_tok) == (100, 50)
    assert total == pytest.approx(0.002)
    assert meta_safe["cash_cost_usd"] == 0.0
    assert meta_safe["imputed_cost_usd"] == pytest.approx(0.002)

    cloud = {"prompt_tokens": 10, "completion_tokens": 10, "cost_per_1k": 0.004}
    _, _, total, _, meta_safe = parse_meta_cost(cloud, "openai/gpt-5.5", lambda *a: 0.0)
    assert total == pytest.approx(0.004)
    assert meta_safe["imputed_cost_usd"] == 0.0


def test_fallback_prices_cover_current_candidates(monkeypatch):
    monkeypatch.setattr(pricing, "_LAST_UPDATE", float("inf"))
    monkeypatch.setattr(pricing, "_PRICING_CACHE", {})
    # 1k in + 1k out a preços de lista (2026-09-18)
    assert pricing.get_model_cost("anthropic/claude-fable-5", 1000, 1000) == pytest.approx(0.060)
    assert pricing.get_model_cost("openai/gpt-5.5-2026-04-23", 1000, 1000) == pytest.approx(0.035)
    assert pricing.get_model_cost("anthropic/claude-sonnet-5", 1000, 1000) == pytest.approx(0.012)
    assert pricing.get_model_cost("anthropic/claude-opus-4-8", 1000, 1000) == pytest.approx(0.030)
    assert pricing.get_model_cost("ollama/gemma3:4b", 1000, 1000) == 0.0  # caixa; ocupação é imputada à parte
