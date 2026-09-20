# Objective: The retention loops must report what they delete and why they failed.
"""``query_log`` was deleted by a loop with ``except Exception: pass``.

It guarded the table the whole empirical analysis rests on, and it logged
nothing at all: it could fail every day for a year and the only symptom would
be a filling disk. Its sibling, guarding ``ema_history_log``, already logged the
row count and the error — the two loops were near-identical except in how
loudly they failed, and the silent one held the primary data.

They now share this loop, which also gains the property the analysis needs:
``days <= 0`` disables the deletion, so an experiment environment can keep
everything.
"""

import threading

import pytest
from app.services.router_maintenance import retention_loop


class FakeLogger:
    def __init__(self):
        self.infos, self.warnings = [], []

    def info(self, msg):
        self.infos.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)


class FakeConn:
    def __init__(self, rowcount):
        self._rowcount = rowcount
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, statement, params=None):
        self.statements.append((str(statement), params))
        return type("R", (), {"rowcount": self._rowcount})()


class FakeEngine:
    def __init__(self, rowcount=0, fail=False):
        self.conn = FakeConn(rowcount)
        self.fail = fail

    def begin(self):
        if self.fail:
            raise RuntimeError("mariadb em baixo")
        return self.conn


def once() -> threading.Event:
    """A stop event that lets exactly one iteration run."""
    event = threading.Event()
    original = event.wait

    def _wait(timeout=None):
        event.set()
        return original(0)

    event.wait = _wait  # type: ignore[method-assign]
    return event


def run(engine, *, days=7, logger=None, before=None, on_deleted=None):
    logger = logger or FakeLogger()
    retention_loop(
        stop_event=once(),
        table="query_log",
        days=days,
        engine_factory=lambda: engine,
        logger=logger,
        before=before,
        on_deleted=on_deleted,
        interval=0,
    )
    return logger


def test_deleted_rows_are_reported():
    engine = FakeEngine(rowcount=42)
    logger = run(engine)
    assert any("42" in msg for msg in logger.infos)


def test_deleting_nothing_is_not_reported():
    """A daily no-op should not fill the log."""
    assert run(FakeEngine(rowcount=0)).infos == []


def test_the_retention_window_reaches_the_query():
    engine = FakeEngine(rowcount=1)
    run(engine, days=90)
    _, params = engine.conn.statements[0]
    assert params == {"d": 90}


@pytest.mark.parametrize("days", [0, -1])
def test_zero_disables_the_deletion(days):
    """The value for an experiment environment: a month-long run used to lose
    its first three weeks while it was still going."""
    engine = FakeEngine(rowcount=5)
    logger = run(engine, days=days)
    assert engine.conn.statements == []
    assert logger.infos == [] and logger.warnings == []


def test_a_failure_is_logged_instead_of_swallowed():
    logger = run(FakeEngine(fail=True))
    assert logger.warnings and "mariadb em baixo" in logger.warnings[0]


def test_the_before_hook_runs_first():
    """``query_log`` needs its table ensured before the DELETE."""
    order = []
    engine = FakeEngine(rowcount=1)
    run(engine, before=lambda: order.append("before"))
    assert order == ["before"]


def test_a_failing_before_hook_does_not_kill_the_loop():
    logger = run(FakeEngine(rowcount=1), before=lambda: (_ for _ in ()).throw(RuntimeError("ensure falhou")))
    assert logger.warnings and "ensure falhou" in logger.warnings[0]


def test_the_on_deleted_hook_receives_the_count():
    """``ema_history_log`` feeds a Prometheus counter from it."""
    seen = []
    run(FakeEngine(rowcount=7), on_deleted=seen.append)
    assert seen == [7]


def test_the_loop_stops_when_the_event_is_set():
    engine = FakeEngine(rowcount=1)
    stop = threading.Event()
    stop.set()
    retention_loop(
        stop_event=stop, table="query_log", days=7,
        engine_factory=lambda: engine, logger=FakeLogger(), interval=0,
    )
    assert engine.conn.statements == []
