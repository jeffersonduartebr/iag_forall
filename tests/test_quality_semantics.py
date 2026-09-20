# Objective: Tests for the quality-semantics flag and the state namespace that keeps learned history apart.
"""Switching the meaning of a score must not corrupt what was learned from it.

The reward feeds the bandit's Beta posteriors and the shared EMAs. Changing what
``quality`` means without separating the state would keep those distributions
updating from a prior accumulated under another definition — nothing would
error, the numbers would keep moving, and the exploration history would be
quietly wrong.

The property that makes the change safe is narrow and absolute: under the
default semantics the namespace is the **identity**. If it were not, merely
deploying this module would orphan every existing key.
"""

import pytest
from app.services import quality_semantics as qs


@pytest.fixture
def semantics(monkeypatch):
    """Set the active semantics without touching the real settings backend."""

    def _set(value):
        monkeypatch.setattr(qs.settings, "get", lambda key, default=None: value if key == qs.SETTING_KEY else default)

    return _set


# ---------------------------------------------------------------------------
# The identity property
# ---------------------------------------------------------------------------


def test_the_default_semantics_is_the_legacy_one(semantics):
    semantics(None)
    assert qs.current_semantics() == qs.SEMANTICS_RUBRIC_V1


def test_under_the_default_the_namespace_is_the_identity(semantics):
    """Deploying the namespacing must not move a single existing key."""
    semantics(qs.SEMANTICS_RUBRIC_V1)
    assert qs.state_namespace() is None
    for value in ("text", "vision", "ctx:cluster-3", "qualquer-coisa"):
        assert qs.namespaced(value) == value


def test_under_the_formative_semantics_state_is_separated(semantics):
    semantics(qs.SEMANTICS_FORMATIVE_V2)
    assert qs.state_namespace() == qs.SEMANTICS_FORMATIVE_V2
    assert qs.namespaced("text") == "formative_v2:text"


def test_the_two_semantics_never_share_a_key(semantics):
    semantics(qs.SEMANTICS_RUBRIC_V1)
    legacy = qs.namespaced("ctx-7")
    semantics(qs.SEMANTICS_FORMATIVE_V2)
    assert qs.namespaced("ctx-7") != legacy


# ---------------------------------------------------------------------------
# Defensive reading
# ---------------------------------------------------------------------------


def test_an_unknown_value_falls_back_instead_of_raising(semantics):
    """A typo in a setting must not take the routing path down."""
    semantics("semantica-inventada")
    assert qs.current_semantics() == qs.SEMANTICS_RUBRIC_V1
    assert qs.namespaced("text") == "text"


def test_a_settings_backend_that_raises_falls_back(monkeypatch):
    def explode(key, default=None):
        raise RuntimeError("redis fora de servico")

    monkeypatch.setattr(qs.settings, "get", explode)
    assert qs.current_semantics() == qs.SEMANTICS_RUBRIC_V1


def test_whitespace_around_the_value_is_tolerated(semantics):
    semantics("  formative_v2  ")
    assert qs.current_semantics() == qs.SEMANTICS_FORMATIVE_V2


def test_is_formative_reflects_the_flag(semantics):
    semantics(qs.SEMANTICS_RUBRIC_V1)
    assert not qs.is_formative()
    semantics(qs.SEMANTICS_FORMATIVE_V2)
    assert qs.is_formative()


# ---------------------------------------------------------------------------
# Where the namespace is actually applied
# ---------------------------------------------------------------------------


def test_the_bandit_redis_key_is_unchanged_under_the_default(semantics):
    from app import bandits

    semantics(qs.SEMANTICS_RUBRIC_V1)
    assert bandits._ctx_key("ctx-7") == f"{bandits.R_CTX_PREFIX}:ctx-7"


def test_the_bandit_redis_key_separates_under_the_new_semantics(semantics):
    from app import bandits

    semantics(qs.SEMANTICS_FORMATIVE_V2)
    assert bandits._ctx_key("ctx-7") == f"{bandits.R_CTX_PREFIX}:formative_v2:ctx-7"


def test_the_ema_redis_key_is_unchanged_under_the_default(semantics):
    from app.services import ema_store

    semantics(qs.SEMANTICS_RUBRIC_V1)
    assert ema_store._ema_key("text") == "ema:text"


def test_the_ema_redis_key_separates_under_the_new_semantics(semantics):
    from app.services import ema_store

    semantics(qs.SEMANTICS_FORMATIVE_V2)
    assert ema_store._ema_key("text") == "ema:formative_v2:text"


def test_the_bandit_database_context_is_namespaced_on_read(semantics, monkeypatch):
    """bandit_context_stats is keyed by a free-form label, so no schema change."""
    from app import bandits

    seen = []
    monkeypatch.setattr(bandits, "load_stats_from_db", lambda engine, ctx: seen.append(ctx) or {})

    semantics(qs.SEMANTICS_RUBRIC_V1)
    bandits._get_ctx_stats_from_db("ctx-7")
    semantics(qs.SEMANTICS_FORMATIVE_V2)
    bandits._get_ctx_stats_from_db("ctx-7")

    assert seen == ["ctx-7", "formative_v2:ctx-7"]


def test_the_bandit_database_context_is_namespaced_on_write(semantics, monkeypatch):
    from app import bandits

    seen = []
    monkeypatch.setattr(bandits, "upsert_stats_db", lambda engine, updates: seen.extend(updates))

    semantics(qs.SEMANTICS_FORMATIVE_V2)
    bandits._batch_upsert_ctx_db([("ctx-7", "modelo", {"mean": 0.5})])
    assert seen[0][0] == "formative_v2:ctx-7"


def test_the_ema_row_carries_the_semantics():
    from app import router_core

    row = router_core._ema_row(
        "text", "m", {"ema_latency": 1.0, "ema_quality": 8.0, "ema_cost": 0.1, "updates": 3}
    )
    assert row["sem"] in qs.VALID_SEMANTICS


def test_the_ema_upsert_keys_on_the_semantics():
    """Without it the calibrated EMAs would overwrite the rubric ones."""
    from app import router_core

    sql = str(router_core._EMA_UPSERT_SQL)
    assert "semantics" in sql
    assert ":sem" in sql


# ---------------------------------------------------------------------------
# What the bandit learns from
# ---------------------------------------------------------------------------


def rubric(q_tech=10.0, q_calibrado=0.0, status="calibrated", p=1.0):
    return {"q_tech": q_tech, "q_calibrado": q_calibrado, "p_entrega": p, "calibration_status": status}


def quality_with(rubric_payload, value=8.0):
    from app.services.feedback_stages import Quality

    return Quality(value, "judge", rubric_payload)


def test_under_the_legacy_semantics_the_bandit_learns_from_the_rubric_mean(semantics):
    from app.services.feedback_stages import learned_quality

    semantics(qs.SEMANTICS_RUBRIC_V1)
    assert learned_quality(quality_with(rubric())) == 8.0


def test_under_the_formative_semantics_the_bandit_learns_from_the_calibrated_score(semantics):
    """This is the only place the switch changes what is optimised."""
    from app.services.feedback_stages import learned_quality

    semantics(qs.SEMANTICS_FORMATIVE_V2)
    assert learned_quality(quality_with(rubric(q_calibrado=0.0))) == 0.0


def test_an_uncalibrated_row_falls_back_to_the_rubric_mean(semantics):
    """A judged answer must not lose its reward because the usurpation judge was down."""
    from app.services.feedback_stages import learned_quality

    semantics(qs.SEMANTICS_FORMATIVE_V2)
    assert learned_quality(quality_with(rubric(status="unavailable", q_calibrado=None))) == 8.0


def test_a_disabled_calibration_falls_back_too(semantics):
    from app.services.feedback_stages import learned_quality

    semantics(qs.SEMANTICS_FORMATIVE_V2)
    assert learned_quality(quality_with(rubric(status="disabled", q_calibrado=10.0))) == 8.0


def test_a_proxy_quality_without_a_rubric_falls_back(semantics):
    from app.services.feedback_stages import Quality, learned_quality

    semantics(qs.SEMANTICS_FORMATIVE_V2)
    assert learned_quality(Quality(6.0, "bandit_proxy")) == 6.0


def test_the_error_predictor_still_reads_the_uncalibrated_quality():
    """Feeding usurpation into it would make it forecast pedagogical failure.

    The predictor exists to tell the router how much effort a request needs, and
    an answer that is technically fine but hands the solution over is not a
    request that needed more effort.
    """
    import inspect

    from app.services import feedback_stages

    source = inspect.getsource(feedback_stages.judge_quality)
    assert "_learn_from_judgment(risk, value)" in source
    assert "learned_quality" not in source


# ---------------------------------------------------------------------------
# What is written to query_log
# ---------------------------------------------------------------------------


def test_every_row_records_which_semantics_its_quality_carries(semantics):
    from app.services.feedback_stages import FeedbackRequest, _formative_fields

    semantics(qs.SEMANTICS_FORMATIVE_V2)
    fb = FeedbackRequest(
        query="q", answer="a", chosen_model="m", modality="text", latency_s=1.0, cost_val=0.0,
        raw_payload={"detected_complexity": "expert"},
    )
    fields = _formative_fields(quality_with(rubric()), fb)
    assert fields["quality_semantics"] == qs.SEMANTICS_FORMATIVE_V2
    assert fields["q_tech"] == 10.0
    assert fields["q_calibrado"] == 0.0
    assert fields["p_entrega"] == 1.0
    assert fields["detected_complexity"] == "expert"


def test_a_row_without_a_rubric_leaves_the_formative_columns_null(semantics):
    from app.services.feedback_stages import FeedbackRequest, Quality, _formative_fields

    semantics(qs.SEMANTICS_RUBRIC_V1)
    fb = FeedbackRequest(
        query="q", answer="a", chosen_model="m", modality="text", latency_s=1.0, cost_val=0.0
    )
    fields = _formative_fields(Quality(6.0, "bandit_proxy"), fb)
    assert fields["quality_semantics"] == qs.SEMANTICS_RUBRIC_V1
    assert fields["q_tech"] is None
    assert fields["q_calibrado"] is None


def test_the_insert_accepts_and_defaults_the_semantics_column():
    import inspect

    from app.query_service import insert_query_log

    params = inspect.signature(insert_query_log).parameters
    for name in ("quality_semantics", "q_tech", "q_calibrado", "p_entrega", "detected_complexity"):
        assert name in params, name
        assert params[name].default is None
