# Objective: Tests for the usurpation judge: ordinal parsing, median aggregation and the annihilation term.
"""The formative score is a product, and products fail differently from sums.

Two properties carry the whole design. First, no amount of technical quality
offsets full delivery — that is what "non-compensatory" means and it has to be
exact, not approximate. Second, a judge that could not answer must never be
read as "did not usurp": treating an infrastructure failure as good pedagogy
would quietly reward outages.
"""

import asyncio

import pytest
from app.services.judge_rubric import DEFAULT_RUBRIC_WEIGHTS, TECH_DIMENSIONS, weighted_quality
from app.services.judge_usurpation import (
    DEFAULT_GAMMA,
    DELIVERY_LEVELS,
    aggregate_delivery,
    annihilation_factor,
    build_usurpation_prompt,
    calibrated_quality,
    parse_delivery_level,
    score_usurpation,
)


def reply(level, evidence="trecho"):
    return f'<reasoning>porque sim</reasoning><entrega>{{"nivel_entrega": {level}, "evidencia": "{evidence}"}}</entrega>'


# ---------------------------------------------------------------------------
# The annihilation term
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p,expected", [(0.0, 1.0), (0.25, 0.984375), (0.5, 0.875), (0.75, 0.578125), (1.0, 0.0)]
)
def test_the_annihilation_term_matches_the_declared_levels(p, expected):
    assert annihilation_factor(p) == pytest.approx(expected)


def test_full_delivery_annihilates_a_perfect_technical_score():
    """The property the whole design exists for."""
    assert calibrated_quality(10.0, 1.0) == 0.0


def test_a_hint_is_almost_free():
    """Penalising the first hint would push models to refuse to help at all."""
    assert calibrated_quality(9.0, 0.25) == pytest.approx(8.86, abs=0.01)


def test_the_penalty_is_flat_near_zero_and_steep_near_one():
    """gamma=3 makes the derivative vanish at p=0, which is the intended shape.

    The claim is about convexity, so the property is the *ratio* between the two
    deciles: the last tenth of delivery costs more than a hundred times what the
    first tenth costs (0.271 against 0.001).
    """
    first_decile = annihilation_factor(0.0) - annihilation_factor(0.1)
    last_decile = annihilation_factor(0.9) - annihilation_factor(1.0)
    assert first_decile < 0.01
    assert last_decile / first_decile > 100


def test_half_the_score_is_lost_only_beyond_level_three():
    half = 2 ** (-1 / 3)
    assert annihilation_factor(half) == pytest.approx(0.5)
    assert 0.75 < half < 1.0


def test_an_unknown_delivery_leaves_the_technical_score_untouched():
    assert annihilation_factor(None) == 1.0
    assert calibrated_quality(7.5, None) == pytest.approx(7.5)


def test_out_of_range_values_are_clamped():
    assert annihilation_factor(-1.0) == 1.0
    assert annihilation_factor(5.0) == 0.0


def test_gamma_is_configurable():
    assert annihilation_factor(0.5, gamma=1.0) == pytest.approx(0.5)
    assert annihilation_factor(0.5, gamma=DEFAULT_GAMMA) == pytest.approx(0.875)


# ---------------------------------------------------------------------------
# Q_tech
# ---------------------------------------------------------------------------


def test_q_tech_renormalises_clarity_and_accuracy_over_their_own_weights():
    """(0.3*q_c + 0.5*q_a) / 0.8, which is the specified formula."""
    scores = {"clareza": 6.0, "acuracia": 10.0, "alinhamento": 0.0}
    expected = (0.3 * 6.0 + 0.5 * 10.0) / 0.8
    assert weighted_quality(scores, DEFAULT_RUBRIC_WEIGHTS, TECH_DIMENSIONS) == pytest.approx(expected)


def test_q_tech_ignores_the_alignment_dimension_entirely():
    low = {"clareza": 8.0, "acuracia": 8.0, "alinhamento": 0.0}
    high = {"clareza": 8.0, "acuracia": 8.0, "alinhamento": 10.0}
    assert weighted_quality(low, DEFAULT_RUBRIC_WEIGHTS, TECH_DIMENSIONS) == weighted_quality(
        high, DEFAULT_RUBRIC_WEIGHTS, TECH_DIMENSIONS
    )


def test_the_full_rubric_remains_the_default():
    """Every existing caller of weighted_quality must be unaffected."""
    scores = {"clareza": 8.0, "acuracia": 8.0, "alinhamento": 0.0}
    assert weighted_quality(scores, DEFAULT_RUBRIC_WEIGHTS) == pytest.approx(0.3 * 8 + 0.5 * 8)


def test_the_old_alignment_penalty_was_compensable_and_the_new_one_is_not():
    """The comparison that justifies replacing a sum with a product."""
    perfect_tech = {"clareza": 10.0, "acuracia": 10.0, "alinhamento": 0.0}
    old_score = weighted_quality(perfect_tech, DEFAULT_RUBRIC_WEIGHTS)
    new_score = calibrated_quality(weighted_quality(perfect_tech, DEFAULT_RUBRIC_WEIGHTS, TECH_DIMENSIONS), 1.0)
    assert old_score == pytest.approx(8.0)  # lost only two points out of ten
    assert new_score == 0.0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level,expected", sorted(DELIVERY_LEVELS.items()))
def test_every_ordinal_level_parses(level, expected):
    assert parse_delivery_level(reply(level)) == pytest.approx(expected)


def test_a_fractional_answer_is_accepted_as_a_fallback():
    assert parse_delivery_level('<entrega>{"p_entrega": 0.4}</entrega>') == pytest.approx(0.4)


def test_a_bare_json_object_is_accepted():
    assert parse_delivery_level('{"nivel_entrega": 2}') == pytest.approx(0.5)


@pytest.mark.parametrize(
    "text", [None, "", "nao sei responder", "<entrega>texto solto</entrega>", '<entrega>{"outro": 1}</entrega>']
)
def test_an_unreadable_reply_is_none_not_zero(text):
    """Zero would mean 'did not usurp' — a free pass for a broken judge."""
    assert parse_delivery_level(text) is None


def test_a_level_outside_the_scale_is_rejected():
    assert parse_delivery_level(reply(9)) is None


def test_a_fraction_out_of_range_is_clamped():
    assert parse_delivery_level('<entrega>{"p_entrega": 2.5}</entrega>') == 1.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_the_median_resists_one_outlying_judge():
    """With the mean, a lone level-4 would annihilate a score the others passed."""
    assert aggregate_delivery([0.25, 0.25, 1.0]) == pytest.approx(0.25)


def test_two_judges_agreeing_give_their_value():
    assert aggregate_delivery([0.5, 0.5]) == pytest.approx(0.5)


def test_aggregating_nothing_gives_none():
    assert aggregate_delivery([]) is None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def rater(mapping):
    async def _rate(model):
        value = mapping[model]
        if isinstance(value, Exception):
            raise value
        return value, {"model": model}

    return _rate


def test_two_agreeing_judges_need_no_referee():
    called = []

    def meta():
        called.append(1)
        return "referee"

    value, info = asyncio.run(
        score_usurpation(["a", "b"], rater({"a": 0.5, "b": 0.5}), meta_model=meta)
    )
    assert value == pytest.approx(0.5)
    assert info["n_judges"] == 2
    assert not called


def test_a_wide_disagreement_calls_the_referee():
    value, info = asyncio.run(
        score_usurpation(
            ["a", "b"],
            rater({"a": 0.0, "b": 1.0, "referee": 0.75}),
            meta_model=lambda: "referee",
        )
    )
    assert info["n_judges"] == 3
    assert value == pytest.approx(0.75)


def test_a_failed_judge_is_dropped_not_counted_as_zero():
    value, info = asyncio.run(
        score_usurpation(["a", "b"], rater({"a": RuntimeError("502"), "b": 1.0}))
    )
    assert info["n_judges"] == 1
    assert value == pytest.approx(1.0)


def test_an_unreadable_judge_is_dropped_too():
    value, info = asyncio.run(score_usurpation(["a", "b"], rater({"a": None, "b": 0.25})))
    assert info["n_judges"] == 1
    assert value == pytest.approx(0.25)


def test_all_judges_failing_gives_no_verdict():
    value, info = asyncio.run(
        score_usurpation(["a", "b"], rater({"a": None, "b": RuntimeError("x")}))
    )
    assert value is None
    assert info["aggregate"] == "unavailable"


def test_a_referee_that_fails_leaves_the_two_original_judges():
    value, info = asyncio.run(
        score_usurpation(
            ["a", "b"],
            rater({"a": 0.0, "b": 1.0, "referee": RuntimeError("timeout")}),
            meta_model=lambda: "referee",
        )
    )
    assert info["n_judges"] == 2
    assert value == pytest.approx(0.5)


def test_each_valid_rating_is_reported():
    seen = []
    asyncio.run(
        score_usurpation(
            ["a", "b"], rater({"a": 0.25, "b": 0.5}), on_rating=lambda m, v, i: seen.append((m, v))
        )
    )
    assert seen == [("a", 0.25), ("b", 0.5)]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_the_prompt_asks_about_delivery_not_correctness():
    prompt = build_usurpation_prompt("Quanto e 15% de 180?", "E 27.")
    assert "ENTREGOU" in prompt and "GUIOU" in prompt
    assert "nunca a correção" in prompt or "nunca a correcao" in prompt


def test_the_prompt_describes_all_five_levels():
    prompt = build_usurpation_prompt("q", "a")
    for level in DELIVERY_LEVELS:
        assert f"{level} -" in prompt


def test_the_scaffolding_path_anchors_the_judgement_when_available():
    """Without a reference path the verdict is an impression; with one it is a check."""
    prompt = build_usurpation_prompt("q", "a", ["Identificar a formula", "Substituir os valores"])
    assert "CAMINHO IDEAL DE ANDAIME" in prompt
    assert "1. Identificar a formula" in prompt
    assert "2. Substituir os valores" in prompt


def test_without_a_path_the_prompt_makes_no_reference_to_one():
    assert "CAMINHO IDEAL" not in build_usurpation_prompt("q", "a")


def test_the_prompt_warns_that_length_is_not_delivery():
    prompt = build_usurpation_prompt("q", "a")
    assert "longa" in prompt and "curta" in prompt


# ---------------------------------------------------------------------------
# Remaining defensive branches
# ---------------------------------------------------------------------------


def test_a_non_numeric_level_falls_through_to_the_fraction():
    """A judge that wrote a word where a number belonged is not silently zero."""
    assert parse_delivery_level('<entrega>{"nivel_entrega": "alto"}</entrega>') is None


def test_a_non_numeric_level_with_a_usable_fraction_is_recovered():
    text = '<entrega>{"nivel_entrega": "alto", "p_entrega": 0.6}</entrega>'
    assert parse_delivery_level(text) == pytest.approx(0.6)


def test_a_non_numeric_fraction_is_rejected():
    assert parse_delivery_level('<entrega>{"p_entrega": "muito"}</entrega>') is None


def test_a_nan_fraction_is_rejected():
    assert parse_delivery_level('<entrega>{"p_entrega": NaN}</entrega>') is None


def test_no_referee_is_called_when_none_is_configured():
    """Without a meta-judge the two disagreeing judges are simply aggregated."""
    value, info = asyncio.run(score_usurpation(["a", "b"], rater({"a": 0.0, "b": 1.0})))
    assert info["n_judges"] == 2
    assert value == pytest.approx(0.5)


def test_a_referee_resolver_that_returns_nothing_is_respected():
    value, info = asyncio.run(
        score_usurpation(["a", "b"], rater({"a": 0.0, "b": 1.0}), meta_model=lambda: None)
    )
    assert info["n_judges"] == 2


def test_a_referee_that_answers_unreadably_is_dropped():
    value, info = asyncio.run(
        score_usurpation(
            ["a", "b"],
            rater({"a": 0.0, "b": 1.0, "referee": None}),
            meta_model=lambda: "referee",
        )
    )
    assert info["n_judges"] == 2


def test_three_judges_are_not_sent_to_a_referee():
    """The referee exists to break a tie between two, not to add a fourth opinion."""
    called = []
    value, info = asyncio.run(
        score_usurpation(
            ["a", "b", "c"],
            rater({"a": 0.0, "b": 1.0, "c": 0.5}),
            meta_model=lambda: called.append(1) or "referee",
        )
    )
    assert info["n_judges"] == 3
    assert not called


def test_the_dispersion_of_the_judges_is_reported():
    _, info = asyncio.run(score_usurpation(["a", "b"], rater({"a": 0.25, "b": 1.0})))
    assert info["dispersion"] == pytest.approx(0.75)


def test_the_referees_rating_is_also_reported():
    """The caller persists every judge's rating, the referee included."""
    seen = []
    asyncio.run(
        score_usurpation(
            ["a", "b"],
            rater({"a": 0.0, "b": 1.0, "referee": 0.5}),
            meta_model=lambda: "referee",
            on_rating=lambda m, v, i: seen.append(m),
        )
    )
    assert seen == ["a", "b", "referee"]


# ---------------------------------------------------------------------------
# Calibration applied to a rubric payload
# ---------------------------------------------------------------------------


RUBRIC_PAYLOAD = {
    "quality": 8.0,
    "dimensions": {"clareza": 10.0, "acuracia": 10.0, "alinhamento": 0.0},
    "n_judges": 2,
    "aggregate": "mean",
}


def test_calibration_leaves_the_rubric_quality_untouched():
    """quality still means what the cache gate and the error predictor expect."""
    from app.services.judge_usurpation import apply_calibration

    out = apply_calibration(RUBRIC_PAYLOAD, 1.0, {}, DEFAULT_RUBRIC_WEIGHTS)
    assert out["quality"] == 8.0


def test_calibration_adds_the_formative_fields_beside_it():
    from app.services.judge_usurpation import apply_calibration

    out = apply_calibration(RUBRIC_PAYLOAD, 1.0, {"n_judges": 2}, DEFAULT_RUBRIC_WEIGHTS)
    assert out["q_tech"] == pytest.approx(10.0)
    assert out["q_calibrado"] == 0.0
    assert out["p_entrega"] == 1.0
    assert out["calibration_status"] == "calibrated"


def test_an_unavailable_delivery_leaves_the_calibrated_score_equal_to_q_tech():
    from app.services.judge_usurpation import apply_calibration

    out = apply_calibration(RUBRIC_PAYLOAD, None, {}, DEFAULT_RUBRIC_WEIGHTS)
    assert out["calibration_status"] == "unavailable"
    assert out["q_calibrado"] == pytest.approx(out["q_tech"])


def test_a_disabled_judge_is_labelled_as_such_not_as_a_failure():
    """'disabled' and 'unavailable' must be distinguishable in the report."""
    from app.services.judge_usurpation import apply_calibration

    out = apply_calibration(RUBRIC_PAYLOAD, None, {}, DEFAULT_RUBRIC_WEIGHTS, enabled=False)
    assert out["calibration_status"] == "disabled"


def test_a_payload_without_dimensions_falls_back_to_its_quality():
    from app.services.judge_usurpation import apply_calibration

    out = apply_calibration({"quality": 7.0}, 0.0, {}, DEFAULT_RUBRIC_WEIGHTS)
    assert out["q_tech"] == 7.0


# ---------------------------------------------------------------------------
# Integration, and the flag that keeps it off
# ---------------------------------------------------------------------------


def test_with_the_flag_off_no_extra_judge_is_called(monkeypatch):
    """The acceptance criterion: flag off means the judging path of today."""
    from app import judges

    called = []
    monkeypatch.setattr(judges, "_safe_setting_float", lambda key, default: 0.0 if "USURPATION" in key else default)
    monkeypatch.setattr(judges, "score_usurpation", lambda *a, **k: called.append(1))

    out = asyncio.run(
        judges._calibrate_delivery(RUBRIC_PAYLOAD, "q", "a", [], DEFAULT_RUBRIC_WEIGHTS, None)
    )
    assert not called
    assert out["calibration_status"] == "disabled"
    assert out["quality"] == RUBRIC_PAYLOAD["quality"]


def test_with_the_flag_on_the_usurpation_judge_runs(monkeypatch):
    from app import judges

    async def fake_score(models, rate, **kwargs):
        return 0.75, {"n_judges": 2}

    monkeypatch.setattr(judges, "_safe_setting_float", lambda key, default: 1.0 if key.endswith("ENABLED") else default)
    monkeypatch.setattr(judges, "score_usurpation", fake_score)

    out = asyncio.run(
        judges._calibrate_delivery(RUBRIC_PAYLOAD, "q", "a", [], DEFAULT_RUBRIC_WEIGHTS, None)
    )
    assert out["p_entrega"] == 0.75
    assert out["q_calibrado"] == pytest.approx(10.0 * 0.578125, abs=0.01)


def test_the_scaffolding_path_reaches_the_prompt(monkeypatch):
    """The link between the corpus and the judge: the reference path."""
    from app import judges

    seen = {}

    async def fake_score(models, rate, **kwargs):
        await rate("juiz")
        return 0.5, {}

    async def fake_rate(call_model, model, prompt, temperature, max_tokens):
        seen["prompt"] = prompt
        return 0.5, {}

    monkeypatch.setattr(judges, "_safe_setting_float", lambda key, default: 1.0 if key.endswith("ENABLED") else default)
    monkeypatch.setattr(judges, "score_usurpation", fake_score)
    monkeypatch.setattr(judges, "rate_usurpation", fake_rate)

    asyncio.run(
        judges._calibrate_delivery(
            RUBRIC_PAYLOAD, "q", "a", [], DEFAULT_RUBRIC_WEIGHTS, ["Identificar a formula", "Substituir"]
        )
    )
    assert "CAMINHO IDEAL DE ANDAIME" in seen["prompt"]
    assert "1. Identificar a formula" in seen["prompt"]


# ---------------------------------------------------------------------------
# One judge call
# ---------------------------------------------------------------------------


def fake_model(result):
    async def _call(**kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    return _call


def test_one_judge_call_returns_the_level_and_the_metadata():
    from app.services.judge_usurpation import rate_usurpation

    level, meta = asyncio.run(
        rate_usurpation(fake_model((reply(2), {"latency": 1.5})), "m", "prompt", 0.0, 512)
    )
    assert level == pytest.approx(0.5)
    assert meta["latency"] == 1.5


def test_a_provider_failure_gives_no_level_and_empty_metadata():
    from app.services.judge_usurpation import rate_usurpation

    level, meta = asyncio.run(
        rate_usurpation(fake_model(RuntimeError("502")), "m", "prompt", 0.0, 512)
    )
    assert level is None
    assert meta == {}


def test_an_unreadable_reply_gives_no_level_but_keeps_the_metadata():
    """The call cost money and took time; that is still worth recording."""
    from app.services.judge_usurpation import rate_usurpation

    level, meta = asyncio.run(
        rate_usurpation(fake_model(("texto solto", {"latency": 2.0})), "m", "prompt", 0.0, 512)
    )
    assert level is None
    assert meta["latency"] == 2.0


def test_non_dict_metadata_is_normalised_to_a_dict():
    from app.services.judge_usurpation import rate_usurpation

    _, meta = asyncio.run(rate_usurpation(fake_model((reply(0), None)), "m", "p", 0.0, 512))
    assert meta == {}
