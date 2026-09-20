# Objective: A feedback task that persisted nothing must not report SUCCESS.
"""The loop swallowed everything and the Celery task finished SUCCESS.

With MariaDB down, every ``/query`` still answered 200, the task finished
successfully, and **no** ``query_log`` row was written, no bandit posterior was
persisted, no ``ema_history`` row appeared. The learning loop stopped in
silence; the only signal was a counter nobody had an alert on.

The distinction the fix draws: a failure to *write the row* fails the task,
because otherwise nothing observes it. An unexpected error in an earlier stage
stays tolerated, because propagating it puts the task into retry and repeating
the pipeline pays the judges a second time for the same row.
"""

import pytest
from app.services.feedback_stages import FeedbackPersistError, FeedbackRequest, Quality, persist_log


class _Metric:
    def labels(self, **kwargs):
        return self

    def inc(self, *a):
        return None

    def set(self, *a):
        return None


def deps_with(insert):
    logged = []
    return logged, {
        "insert_query_log": insert,
        "logger": type("L", (), {"error": staticmethod(lambda *a, **k: logged.append(a)),
                                 "warning": staticmethod(lambda *a, **k: None),
                                 "info": staticmethod(lambda *a, **k: None)})(),
        "FEEDBACK_TASK_FAILURES": _Metric(),
        "ROUTER_QUALITY_AVG": _Metric(),
        "ROUTER_LOCAL_USAGE_RATIO": _Metric(),
        "ROUTER_ROUTING_MATRIX": _Metric(),
    }


def a_request():
    return FeedbackRequest(
        query="q", answer="a", chosen_model="m", modality="text",
        latency_s=1.0, cost_val=0.0, raw_payload={},
    )


def a_risk():
    from app.services.feedback_stages import ErrorRisk

    return ErrorRisk(
        predicted_error_prob=0.1, query_embedding=None, model_stats={}, predictor=None
    )


def test_a_failed_insert_raises_instead_of_being_swallowed():
    def _boom(**kwargs):
        raise RuntimeError("mariadb em baixo")

    logged, deps = deps_with(_boom)
    with pytest.raises(FeedbackPersistError, match="mariadb em baixo"):
        persist_log(deps, a_request(), Quality(6.0, "bandit_proxy"), False, a_risk(), 0.5)
    assert logged, "a perda da linha não foi registada"


def test_a_successful_insert_raises_nothing():
    rows = []
    _, deps = deps_with(lambda **kwargs: rows.append(kwargs))
    persist_log(deps, a_request(), Quality(6.0, "bandit_proxy"), False, a_risk(), 0.5)
    assert len(rows) == 1


def test_the_error_is_a_runtime_error_subclass():
    """Existing `except Exception` handlers upstream keep working."""
    assert issubclass(FeedbackPersistError, RuntimeError)


def test_the_celery_task_does_not_retry_a_persistence_failure():
    """The judges already ran and were paid for; repeating buys the row twice."""
    import inspect

    from app import tasks

    source = inspect.getsource(tasks.task_process_feedback)
    persist_branch = source.index("except FeedbackPersistError")
    generic_branch = source.index("except Exception")
    assert persist_branch < generic_branch, "o ramo específico tem de vir primeiro"
    # No ramo específico, `raise` nu — sem self.retry.
    branch = source[persist_branch:generic_branch]
    assert "self.retry" not in branch
    assert "raise" in branch
