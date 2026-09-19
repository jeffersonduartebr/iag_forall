# Objective: Property-based tests (hypothesis) for reward, rubric, pricing, reliability and EMA math.
"""Invariants of the scoring functions, checked over generated inputs.

Example-based tests pin specific values; these pin the contracts the router
relies on (ranges, monotonicity, weights summing to 1, linearity) for any input.
"""

import math
from unittest.mock import patch

import pytest
from app.services import ema_store, judge_rubric, query_reliability, reward
from app.utils import pricing
from hypothesis import given
from hypothesis import strategies as st

unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
score10 = st.floats(min_value=0.0, max_value=10.0, allow_nan=False)
latency = st.floats(min_value=0.0, max_value=600.0, allow_nan=False)
cost = st.one_of(st.none(), st.floats(min_value=0.0, max_value=10.0, allow_nan=False))
positive = st.floats(min_value=1e-6, max_value=1e3, allow_nan=False)
weights3 = st.tuples(positive, positive, positive).map(lambda w: tuple(x / sum(w) for x in w))
any_float = st.floats(allow_nan=True, allow_infinity=True)


def _reward(weights, q, lat, c):
    with (
        patch.object(reward, "load_reward_weights", lambda modality="text": (weights, "test")),
        patch.object(reward, "_setting_float", lambda key, default: default),
    ):
        return reward.compute_reward("m", q, lat, c)


# ---------------------------------------------------------------- recompensa


@given(raw=st.tuples(*[st.floats(min_value=0.0, max_value=1e4, allow_nan=False)] * 6), floor=unit)
def test_reward_weights_sum_to_one_and_respect_floor(raw, floor):
    shares = reward.derive_reward_weights(*raw, min_share=floor)
    assert sum(shares) == pytest.approx(1.0)
    effective_floor = min(floor, 1.0 / 3.0)
    assert all(s >= effective_floor - 1e-9 for s in shares)


@given(weights=weights3, q=score10, lat=latency, c=cost)
def test_reward_in_unit_interval(weights, q, lat, c):
    assert 0.0 <= _reward(weights, q, lat, c) <= 1.0


@given(weights=weights3, q1=score10, q2=score10, lat=latency, c=cost)
def test_reward_monotonic_in_quality(weights, q1, q2, lat, c):
    lo, hi = sorted((q1, q2))
    assert _reward(weights, lo, lat, c) <= _reward(weights, hi, lat, c) + 1e-12


@given(weights=weights3, q=score10, l1=latency, l2=latency, c=cost)
def test_reward_non_increasing_in_latency(weights, q, l1, l2, c):
    fast, slow = sorted((l1, l2))
    assert _reward(weights, q, slow, c) <= _reward(weights, q, fast, c) + 1e-12


@given(
    weights=weights3,
    q=score10,
    lat=latency,
    c1=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    c2=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
)
def test_reward_non_increasing_in_cost(weights, q, lat, c1, c2):
    cheap, pricey = sorted((c1, c2))
    assert _reward(weights, q, lat, pricey) <= _reward(weights, q, lat, cheap) + 1e-12


@given(c=st.floats(min_value=0.0, max_value=1e3, allow_nan=False), base=positive)
def test_cost_score_bounded(c, base):
    assert reward.COST_SCORE_FLOOR <= reward.cost_score(c, base) <= 1.0


@given(prices=st.lists(st.tuples(positive, positive), min_size=1, max_size=20), share=unit)
def test_cost_baseline_between_extremes(prices, share):
    blended = [reward.blended_cost_per_1k(i, o, share) for i, o in prices]
    baseline = reward.calibrate_cost_baseline(prices, share)
    assert min(blended) * (1 - 1e-9) <= baseline <= max(blended) * (1 + 1e-9)


@given(
    total=st.floats(min_value=0.0, max_value=100.0, allow_nan=False), p=st.integers(0, 10**6), c=st.integers(0, 10**6)
)
def test_cost_per_1k_from_total(total, p, c):
    value = reward.cost_per_1k_from_total(total, p, c)
    if p + c == 0:
        assert value is None
    else:
        assert value == pytest.approx(total / (p + c) * 1000.0)


# ---------------------------------------------------------------- rubrica


@given(values=st.tuples(any_float, any_float, any_float))
def test_rubric_scores_clamped(values):
    text = '{"clareza": %r, "acuracia": %r, "alinhamento": %r}' % tuple(str(v) for v in values)
    scores = judge_rubric.parse_rubric_scores(text)
    if scores is not None:
        assert set(scores) == set(judge_rubric.RUBRIC_DIMENSIONS)
        assert all(0.0 <= s <= 10.0 for s in scores.values())


@given(scores=st.tuples(score10, score10, score10), weights=st.tuples(positive, positive, positive))
def test_weighted_quality_between_dimensions(scores, weights):
    dims = judge_rubric.RUBRIC_DIMENSIONS
    q = judge_rubric.weighted_quality(dict(zip(dims, scores)), dict(zip(dims, weights)))
    assert min(scores) - 1e-9 <= q <= max(scores) + 1e-9


@given(a=st.tuples(score10, score10, score10), b=st.tuples(score10, score10, score10), outlier=score10)
def test_median_of_three_bounded_by_the_agreeing_pair(a, b, outlier):
    dims = judge_rubric.RUBRIC_DIMENSIONS
    ratings = [dict(zip(dims, a)), dict(zip(dims, b)), dict.fromkeys(dims, outlier)]
    out = judge_rubric.combine_ratings(ratings, judge_rubric.DEFAULT_RUBRIC_WEIGHTS, aggregate="median")
    for i, d in enumerate(dims):
        assert min(a[i], b[i]) <= out["dimensions"][d] <= max(a[i], b[i])
    assert out["n_judges"] == 3


# ---------------------------------------------------------------- preços


MODELS = ["openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-opus-4", "gemini/gemini-2.5-flash", "ollama/gemma3"]


@given(
    model=st.sampled_from(MODELS),
    a=st.integers(0, 10**6),
    b=st.integers(0, 10**6),
    c=st.integers(0, 10**6),
    d=st.integers(0, 10**6),
)
def test_model_cost_non_negative_and_additive(model, a, b, c, d):
    with patch.object(pricing, "_LAST_UPDATE", math.inf), patch.object(pricing, "_PRICING_CACHE", {}):
        total = pricing.get_model_cost(model, a + b, c + d)
        parts = pricing.get_model_cost(model, a, c) + pricing.get_model_cost(model, b, d)
    assert total >= 0.0
    assert total == pytest.approx(parts, rel=1e-9, abs=1e-12)


@given(
    s1=st.floats(min_value=0.0, max_value=1e5, allow_nan=False),
    s2=st.floats(min_value=0.0, max_value=1e5, allow_nan=False),
)
def test_local_cost_linear_in_occupancy(s1, s2):
    rates = {"LOCAL_COST_USD_PER_HOUR": 0.36, "LOCAL_COST_PARALLEL_SLOTS": 2.0}
    with patch.object(pricing, "_local_cost_setting", lambda key, default: rates.get(key, default)):
        both = pricing.impute_local_cost(s1 + s2)
        parts = pricing.impute_local_cost(s1) + pricing.impute_local_cost(s2)
    assert both >= 0.0
    assert both == pytest.approx(parts, rel=1e-9, abs=1e-15)


# ---------------------------------------------------------------- confiabilidade


flags = st.fixed_dictionaries(
    {
        "answered": st.booleans(),
        "grounded": st.booleans(),
        "retrieval_used": st.booleans(),
        "fallback_used": st.booleans(),
        "flagged": st.booleans(),
    }
)


@given(u1=unit, u2=unit, kw=flags)
def test_confidence_bounded_and_non_increasing_in_uncertainty(u1, u2, kw):
    lo, hi = sorted((u1, u2))
    s_lo, s_hi = query_reliability.confidence_score(lo, **kw), query_reliability.confidence_score(hi, **kw)
    assert 0.0 <= s_hi <= s_lo <= 1.0


@given(score=unit, answered=st.booleans(), grounded=st.booleans())
def test_band_and_verification_consistent(score, answered, grounded):
    band = query_reliability.confidence_band(score)
    assert band == ("high" if score >= 0.75 else "medium" if score >= 0.45 else "low")
    status = query_reliability.verification_status(answered=answered, grounded=grounded, score=score)
    if not answered:
        assert status == "unsupported"
    elif status == "supported":
        assert grounded and score >= 0.70


# ---------------------------------------------------------------- EMA


@given(samples=st.lists(st.tuples(latency, score10, st.floats(0.0, 1.0)), min_size=1, max_size=50))
def test_ema_stays_within_observed_range(samples):
    entry = None
    for lat, q, c in samples:
        entry = ema_store.next_ema(entry, lat, q, c)
    lats, qs, costs = zip(*samples)
    assert entry["updates"] == len(samples)
    assert min(lats) - 1e-9 <= entry["ema_latency"] <= max(lats) + 1e-9
    assert min(qs) - 1e-9 <= entry["ema_quality"] <= max(qs) + 1e-9
    assert min(costs) - 1e-9 <= entry["ema_cost"] <= max(costs) + 1e-9


@given(updates=st.integers(0, 10), lat=latency, c=st.floats(0.0, 1.0), local=st.booleans(), sota=st.booleans())
def test_routing_latency_cost_uses_ema_only_when_mature(updates, lat, c, local, sota):
    entry = {"ema_latency": lat, "ema_cost": c, "updates": updates}
    got = ema_store.routing_latency_cost(entry, is_local=local, is_sota=sota)
    if updates >= ema_store.MIN_UPDATES_FOR_ROUTING:
        assert got == (lat, c)
    else:
        assert got == ema_store.routing_latency_cost(None, is_local=local, is_sota=sota)
