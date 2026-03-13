# Objective: Test NSGA judge-feedback filtering and meta-optimizer scheduling behavior.
"""Regression tests for NSGA feedback tuning and meta-optimizer runtime mode."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _Conn:
    """Minimal connection stub for sequential SQL fetches."""

    def __init__(self, rows):
        self._rows = list(rows)

    def execute(self, *_args, **_kwargs):
        return SimpleNamespace(fetchone=lambda: self._rows.pop(0))


class _Ctx:
    """Context manager that returns a prepared connection stub."""

    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        return False


def test_tune_weights_from_judge_feedback_ignores_proxy_rows(monkeypatch):
    """Judge-feedback tuning must rely only on judged rows and respect min sample size."""
    from app import nsga_weights_updater as nsga

    monkeypatch.setattr(
        nsga,
        "settings",
        SimpleNamespace(
            get=lambda key, default=None: {
                "JUDGE_FEEDBACK_MIN_SAMPLES": "30",
                "JUDGE_FEEDBACK_ERROR_THRESHOLD": "5.0",
            }.get(key, default),
            NSGA_W_QUALITY=1.0,
            set=lambda *a, **k: pytest.fail("settings.set should not be called for insufficient samples"),
        ),
    )
    monkeypatch.setattr(
        nsga,
        "engine",
        SimpleNamespace(connect=lambda: _Ctx(_Conn([(10, 8), (200,)]))),
    )

    rate_values = []
    monkeypatch.setattr(nsga.JUDGE_FEEDBACK_ERROR_RATE, "set", lambda value: rate_values.append(value))
    proxy_counts = []
    monkeypatch.setattr(nsga.JUDGE_FEEDBACK_PROXY_TOTAL, "inc", lambda value=1: proxy_counts.append(value))
    sampled_counts = []
    monkeypatch.setattr(nsga.JUDGE_FEEDBACK_SAMPLED_TOTAL, "inc", lambda value=1: sampled_counts.append(value))

    nsga.tune_weights_from_judge_feedback()

    assert rate_values[-1] == 0.0
    assert proxy_counts == [200]
    assert sampled_counts == [10]


def test_tune_weights_from_judge_feedback_updates_weight_from_judged_rows(monkeypatch):
    """Judge-feedback tuning should increase quality weight on genuinely judged low-quality samples."""
    from app import nsga_weights_updater as nsga

    updates = []
    monkeypatch.setattr(
        nsga,
        "settings",
        SimpleNamespace(
            get=lambda key, default=None: {
                "JUDGE_FEEDBACK_MIN_SAMPLES": "30",
                "JUDGE_FEEDBACK_ERROR_THRESHOLD": "5.0",
            }.get(key, default),
            NSGA_W_QUALITY=1.2,
            set=lambda key, value, actor=None: updates.append((key, value, actor)),
        ),
    )
    monkeypatch.setattr(
        nsga,
        "engine",
        SimpleNamespace(connect=lambda: _Ctx(_Conn([(40, 20), (5,)]))),
    )
    monkeypatch.setattr(nsga.JUDGE_FEEDBACK_ERROR_RATE, "set", lambda _value: None)
    monkeypatch.setattr(nsga.JUDGE_FEEDBACK_PROXY_TOTAL, "inc", lambda value=1: None)
    monkeypatch.setattr(nsga.JUDGE_FEEDBACK_SAMPLED_TOTAL, "inc", lambda value=1: None)

    nsga.tune_weights_from_judge_feedback()

    assert updates == [("NSGA_W_QUALITY", "1.5", "judge-feedback")]


def test_metaopt_main_defaults_to_scheduler(monkeypatch):
    """The meta optimizer should idle in scheduler mode instead of running one-shot batches."""
    from app import nsga_meta_optimizer as metaopt

    calls = []
    monkeypatch.setenv("META_OPT_MODE", "scheduler")
    monkeypatch.setattr(metaopt, "start_scheduled_optimizer", lambda: calls.append("started"))

    def _stop(_seconds):
        raise SystemExit(0)

    monkeypatch.setattr(metaopt.time, "sleep", _stop)

    with pytest.raises(SystemExit):
        metaopt.main()

    assert calls == ["started"]


def test_metaopt_main_oneshot_path(monkeypatch):
    """The meta optimizer should still support explicit one-shot execution for manual runs."""
    from app import nsga_meta_optimizer as metaopt

    calls = []
    monkeypatch.setenv("META_OPT_MODE", "oneshot")
    monkeypatch.setattr(metaopt, "run_manual_oneshot", lambda: calls.append("oneshot"))

    assert metaopt.main() == 0
    assert calls == ["oneshot"]
