# Objective: Tests for the length-aware latency threshold and the flag that keeps it off by default.
"""The latency deadline now depends on how long the answer was.

Two things have to be true at once. The new behaviour must be correct, and the
old behaviour must be *bit-for-bit unchanged* while the flag is off — because
the reward feeds the bandit's Beta posteriors and an absolute promotion
threshold downstream, and shifting the distribution silently would rewrite the
routing policy without anyone deciding to.
"""

import pytest
from app.services import reward
from app.services.reward import (
    DEFAULT_COMPLETION_TOKENS,
    LATENCY_X0_S,
    MAX_BUDGETED_COMPLETION_TOKENS,
    compute_reward,
    latency_score,
    latency_threshold_s,
)


# ---------------------------------------------------------------------------
# The identity that makes the change safe
# ---------------------------------------------------------------------------


def test_the_default_token_count_reproduces_the_former_threshold_exactly():
    """375 tokens is where the new formula and the old constant coincide."""
    assert latency_threshold_s(DEFAULT_COMPLETION_TOKENS) == LATENCY_X0_S == 20.0


def test_an_unknown_token_count_falls_back_to_that_default():
    assert latency_threshold_s(None) == LATENCY_X0_S


def test_the_half_point_of_the_curve_is_still_the_threshold():
    assert latency_score(20.0, DEFAULT_COMPLETION_TOKENS) == pytest.approx(0.5)
    assert latency_score(7.0, 50) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The threshold itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tokens,expected",
    [(0, 5.0), (25, 6.0), (50, 7.0), (250, 15.0), (375, 20.0), (600, 29.0), (1200, 53.0)],
)
def test_the_threshold_is_ttft_plus_generation_time(tokens, expected):
    assert latency_threshold_s(tokens) == pytest.approx(expected)


def test_a_verbose_answer_cannot_buy_an_unbounded_deadline():
    """Otherwise padding the answer would widen the model's own window."""
    capped = latency_threshold_s(MAX_BUDGETED_COMPLETION_TOKENS)
    assert latency_threshold_s(10**9) == capped


@pytest.mark.parametrize("bad", [-5, "abc", float("nan"), None])
def test_a_nonsensical_token_count_falls_back_instead_of_raising(bad):
    assert latency_threshold_s(bad) == pytest.approx(LATENCY_X0_S)


def test_a_zero_generation_rate_falls_back_instead_of_dividing_by_zero():
    assert latency_threshold_s(375, tokens_per_s=0.0) == pytest.approx(LATENCY_X0_S)


def test_the_budget_and_rate_can_be_overridden():
    assert latency_threshold_s(100, ttft=2.0, tokens_per_s=50.0) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# What the change actually does
# ---------------------------------------------------------------------------


def test_a_short_answer_gets_a_tighter_deadline_than_before():
    """This is the shift that forces the promotion gate to be recalibrated."""
    before = latency_score(10.0, x0=LATENCY_X0_S)
    after = latency_score(10.0, 50)
    assert after < before
    assert before == pytest.approx(0.769, abs=0.01)
    assert after == pytest.approx(0.411, abs=0.01)


def test_a_long_answer_is_no_longer_punished_for_its_length():
    """The requirement: discursive domains stop losing points for being long."""
    before = latency_score(30.0, x0=LATENCY_X0_S)
    after = latency_score(30.0, 1200)
    assert after > before
    assert before == pytest.approx(0.231, abs=0.01)
    assert after == pytest.approx(0.940, abs=0.01)


def test_the_overflow_guard_survives_an_absurd_latency():
    assert latency_score(10**6, 50) == 0.0


# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------


def _settings(monkeypatch, **values):
    monkeypatch.setattr(reward, "_setting_float", lambda key, default: values.get(key, default))


def test_with_the_flag_off_the_token_count_changes_nothing(monkeypatch):
    """The acceptance criterion: flag off means the system of today, unchanged."""
    _settings(monkeypatch, REWARD_DYNAMIC_LATENCY_ENABLED=0.0)
    short = compute_reward("m", 8.0, 10.0, 0.004, "text", completion_tokens=50)
    long = compute_reward("m", 8.0, 10.0, 0.004, "text", completion_tokens=4000)
    none = compute_reward("m", 8.0, 10.0, 0.004, "text")
    assert short == long == none


def test_with_the_flag_on_the_token_count_matters(monkeypatch):
    _settings(monkeypatch, REWARD_DYNAMIC_LATENCY_ENABLED=1.0)
    short = compute_reward("m", 8.0, 10.0, 0.004, "text", completion_tokens=50)
    long = compute_reward("m", 8.0, 10.0, 0.004, "text", completion_tokens=4000)
    assert long > short


def test_with_the_flag_on_the_default_still_matches_the_old_calibration(monkeypatch):
    _settings(monkeypatch, REWARD_DYNAMIC_LATENCY_ENABLED=1.0)
    with_default = compute_reward("m", 8.0, 10.0, 0.004, "text", completion_tokens=375)
    _settings(monkeypatch, REWARD_DYNAMIC_LATENCY_ENABLED=0.0)
    legacy = compute_reward("m", 8.0, 10.0, 0.004, "text")
    assert with_default == pytest.approx(legacy)


def test_the_reward_stays_within_the_unit_interval(monkeypatch):
    """The bandit clamps to [0,1] and builds Beta posteriors on it."""
    _settings(monkeypatch, REWARD_DYNAMIC_LATENCY_ENABLED=1.0)
    for tokens in (0, 50, 375, 8000, 10**7):
        for latency in (0.0, 1.0, 60.0, 600.0):
            value = compute_reward("m", 10.0, latency, 0.0, "text", completion_tokens=tokens)
            assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# Signature compatibility
# ---------------------------------------------------------------------------


def test_completion_tokens_is_keyword_only():
    """Positional callers and their test doubles must keep working untouched."""
    import inspect

    parameter = inspect.signature(compute_reward).parameters["completion_tokens"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_positional_signature_is_unchanged():
    import inspect

    positional = [
        name
        for name, p in inspect.signature(compute_reward).parameters.items()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == ["model", "quality", "latency_s", "cost_per_1k", "modality"]


def test_the_feedback_stage_passes_the_token_count():
    import inspect

    from app.services import feedback_stages

    source = inspect.getsource(feedback_stages.update_bandit)
    assert "completion_tokens=fb.completion_tokens" in source
