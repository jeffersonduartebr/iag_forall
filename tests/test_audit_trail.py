# Objective: A configuration change and a frozen policy must leave a durable trace.
"""Two gaps that make an experimental result irreproducible.

``settings.set`` already took ``actor`` and ``source`` from every caller — the
admin API, the NSGA-II updater, the OpenRouter explorer, the eval feedback loop
— and threw them away. Nothing recorded who changed which setting, when, or
what it was before, so there was no way to know which configuration was active
at the moment a ``query_log`` row was written, nor to see that someone had
changed it halfway through a run.

And the frozen-policy snapshot lived only in Redis under a 24-hour TTL, read by
no analysis path at all: a day after an eval run, the weights it ran under were
simply gone.
"""

import pytest

from app import settings_dynamic as sd


@pytest.fixture
def captured_audit(monkeypatch):
    events = []
    monkeypatch.setattr("app.roadmap_features.log_audit_event", lambda **kw: events.append(kw))
    return events


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["OPENROUTER_API_KEY", "ADMIN_TOKEN", "JWT_SECRET", "DB_PASSWORD", "DB_PASS"]
)
def test_a_secret_value_never_enters_the_audit_table(key):
    """Knowing the key changed is the point; the secret itself would only make
    the audit table a second place to leak it from."""
    assert sd._redact(key, "sk-valor-real") == "<redigido>"


@pytest.mark.parametrize("key", ["NSGA_W_QUALITY", "BANDIT_EPSILON", "CANDIDATE_MODELS_LIST"])
def test_an_ordinary_value_is_recorded(key):
    assert sd._redact(key, "0.42") == "0.42"


def test_a_missing_previous_value_stays_none():
    assert sd._redact("NSGA_W_QUALITY", None) is None


def test_a_long_value_is_truncated():
    assert len(sd._redact("CANDIDATE_MODELS_LIST", "x" * 5000)) == 512


# ---------------------------------------------------------------------------
# What is recorded
# ---------------------------------------------------------------------------


def test_a_change_records_actor_source_and_the_previous_value(captured_audit, monkeypatch):
    monkeypatch.setattr(sd, "_get_from_db", lambda key: "0.10")
    sd._audit_setting_change("BANDIT_EPSILON", "0.10", "0.25", "api", "admin")

    event = captured_audit[-1]
    assert event["actor"] == "api"
    assert event["action"] == "settings.set"
    assert event["resource"] == "BANDIT_EPSILON"
    assert event["metadata"]["source"] == "admin"
    assert event["metadata"]["previous"] == "0.10"
    assert event["metadata"]["value"] == "0.25"
    assert event["metadata"]["changed"] is True


def test_a_no_op_write_is_still_recorded_but_marked(captured_audit):
    """An operator re-applying the same value is a different event from a change."""
    sd._audit_setting_change("BANDIT_EPSILON", "0.25", "0.25", "api", "admin")
    assert captured_audit[-1]["metadata"]["changed"] is False


def test_an_unavailable_audit_table_does_not_block_the_change(monkeypatch):
    """The change that matters most is the one disabling a broken model."""

    def _boom(**kwargs):
        raise RuntimeError("mariadb em baixo")

    monkeypatch.setattr("app.roadmap_features.log_audit_event", _boom)
    sd._audit_setting_change("BANDIT_EPSILON", "0.1", "0.2", "api", "admin")  # não levanta


def test_set_calls_the_audit(monkeypatch):
    """Wiring check: the parameters existed and were being dropped."""
    import inspect

    source = inspect.getsource(sd.DynamicSettings.set)
    assert "_audit_setting_change(key, previous, value, actor, source)" in source
    assert "previous = _get_from_db(key)" in source


# ---------------------------------------------------------------------------
# Frozen policy
# ---------------------------------------------------------------------------


def test_the_eval_run_records_the_policy_it_ran_under():
    """Redis held it for 24 hours and no analysis path ever read it."""
    import inspect

    from app import tasks

    source = inspect.getsource(tasks.task_execute_eval_run)
    assert 'summary["frozen_policy"] = applied_policy' in source


def test_an_unfrozen_run_is_recorded_as_such():
    import inspect

    from app import tasks

    source = inspect.getsource(tasks.task_execute_eval_run)
    assert '{"active": False}' in source
