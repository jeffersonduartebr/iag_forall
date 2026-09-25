# Objective: Coverage for the Bayesian (Optuna) meta-optimizer of NSGA-II hyperparameters.
"""nsga_meta_optimizer: persistence, objective, scheduled/one-shot runs and the scheduler loop."""

import sys
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app import nsga_meta_optimizer as metaopt

PARAMS = dict(N_pop="8", N_gen=5, cxpb=0.7, mutpb="0.1", eta_c=10, eta_m=20.0)


def _engine(calls, fail=False):
    class _Conn:
        def execute(self, stmt, params=None):
            calls.append(params)

    @contextmanager
    def begin():
        if fail:
            raise SQLAlchemyError("db down")
        yield _Conn()

    return SimpleNamespace(begin=begin)


def test_db_engine_comes_from_the_shared_pool(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(metaopt, "get_engine", lambda: sentinel)
    assert metaopt._db_engine() is sentinel


def test_ensure_table_and_save_result(monkeypatch):
    calls = []
    monkeypatch.setattr(metaopt, "_db_engine", lambda: _engine(calls))
    metaopt._ensure_meta_table()
    metaopt.save_result("text", 3, PARAMS, 1.5, 0.25)
    assert calls[0] is None
    assert calls[1] == dict(mod="text", trial=3, np=8, ng=5, cx=0.7, mu=0.1, ec=10.0, em=20.0, mean=1.5, std=0.25)

    monkeypatch.setattr(metaopt, "_db_engine", lambda: _engine(calls, fail=True))
    metaopt._ensure_meta_table()
    metaopt.save_result("text", 4, PARAMS, 1.0, 0.0)  # falhas só são registradas
    assert len(calls) == 2


def test_evaluate_once_posts_to_modality_endpoint(monkeypatch):
    sent = {}

    def post(url, json, timeout):
        sent.update(url=url, json=json)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"efficiency": "2.5"})

    monkeypatch.setattr(metaopt.requests, "post", post)
    assert metaopt.evaluate_once("vision", 8, 5, 0.7, 0.1, 10.0, 20.0) == 2.5
    assert sent["url"].endswith("/run/vision") and sent["json"]["N_pop"] == 8


class _Trial:
    """Stand-in for ``optuna.trial.Trial``: always suggests the upper bound."""

    def __init__(self, number=7):
        self.number, self.params = number, {}

    def suggest_int(self, name, low, high, step=1):
        self.params[name] = high
        return high

    def suggest_float(self, name, low, high):
        self.params[name] = high
        return high


class _Study:
    def __init__(self, **kwargs):
        self.kwargs, self.trials = kwargs, []

    def optimize(self, objective, n_trials, n_jobs):
        for number in range(n_trials):
            trial = _Trial(number)
            self.trials.append((objective(trial), trial))
        self.best_value, self.best_trial = max(self.trials, key=lambda vt: vt[0])


@pytest.fixture(autouse=True)
def fake_optuna(monkeypatch):
    """optuna só existe na imagem do metaopt; o módulo é carregado sob demanda, então um dublê basta."""
    studies = []

    def create_study(**kwargs):
        studies.append(_Study(**kwargs))
        return studies[-1]

    module = SimpleNamespace(create_study=create_study, samplers=SimpleNamespace(TPESampler=lambda seed: ("tpe", seed)))
    monkeypatch.setitem(sys.modules, "optuna", module)
    return studies


def test_objective_averages_reps_and_counts_failures_as_zero(monkeypatch):
    scores = iter([4.0, RuntimeError("nsga down")])
    seen, saved = [], []

    def evaluate(*args):
        seen.append(args)
        value = next(scores)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(metaopt, "evaluate_once", evaluate)
    monkeypatch.setattr(metaopt, "save_result", lambda *a: saved.append(a))
    assert metaopt.build_objective("text", reps=2)(_Trial()) == pytest.approx(2.0)
    # cxpb no máximo (0.95) limita mutpb a 0.05: a soma nunca passa de 1.
    assert seen[0] == pytest.approx(("text", 64, 40, 0.95, 0.05, 40.0, 40.0))
    modality, trial_id, params, mean, std = saved[0]
    assert (modality, trial_id, mean, std) == ("text", 7, 2.0, 2.0)
    assert params["mutpb"] == pytest.approx(0.05)


@pytest.fixture
def optuna_env(monkeypatch):
    monkeypatch.setenv("METAOPT_SCHEDULED_REPS", "1")
    monkeypatch.setenv("METAOPT_REPS", "1")
    monkeypatch.setenv("METAOPT_TRIALS", "1")
    monkeypatch.setattr(metaopt, "evaluate_once", lambda modality, *a: {"text": 3.0, "vision": 2.0}[modality])
    monkeypatch.setattr(metaopt, "save_result", lambda *a: None)
    scripts = []
    monkeypatch.setattr(metaopt.subprocess, "check_call", lambda cmd: scripts.append(cmd))
    return scripts


def test_scheduled_optimization_reports_each_modality(monkeypatch, optuna_env):
    monkeypatch.setattr(type(metaopt_settings()), "META_OPT_SCHEDULED_TRIALS", property(lambda s: 1))
    real_build = metaopt.build_objective

    def build(modality, reps):
        if modality == "multimodal":
            raise ValueError("no multimodal models")
        return real_build(modality, reps)

    monkeypatch.setattr(metaopt, "build_objective", build)
    results = metaopt.run_scheduled_optimization()
    assert results["text"]["best_efficiency"] == 3.0 and results["text"]["best_trial"] == 0
    assert results["vision"]["best_efficiency"] == 2.0 and set(results["vision"]["best_params"]) >= {"N_pop", "cxpb"}
    assert results["multimodal"] == {"error": "no multimodal models"}
    assert optuna_env == [["python", "/app/app/update_nsga_best_params.py"]]


def test_scheduled_and_oneshot_survive_update_script_failure(monkeypatch, optuna_env):
    monkeypatch.setattr(metaopt, "evaluate_once", lambda *a: 1.0)

    def fail(cmd):
        raise OSError("python missing")

    monkeypatch.setattr(metaopt.subprocess, "check_call", fail)
    assert metaopt.run_scheduled_optimization(n_trials=1)["text"]["best_efficiency"] == 1.0
    metaopt.run_manual_oneshot()  # não propaga a falha do script


def test_oneshot_runs_every_modality_then_the_update_script(monkeypatch, optuna_env, fake_optuna):
    monkeypatch.setattr(metaopt, "evaluate_once", lambda *a: 1.0)
    monkeypatch.setenv("METAOPT_TRIALS", "2")
    metaopt.run_manual_oneshot()
    assert optuna_env == [["python", "/app/app/update_nsga_best_params.py"]]
    names = [(s.kwargs["study_name"], s.kwargs["direction"], len(s.trials)) for s in fake_optuna]
    assert names == [(f"nsga_metaopt_{m}", "maximize", 2) for m in ("text", "vision", "multimodal")]


def metaopt_settings():
    from app.settings_dynamic import settings

    return settings


class _Stop(BaseException):
    pass


@pytest.mark.parametrize(
    "enabled, hour, minute, expected_sleeps, runs",
    [
        (False, 3, 0, [3600, 3600], 0),
        (True, 3, 2, [3600], 1),
        (True, 3, 30, [300, 300], 0),
        (True, 9, 0, [300, 300], 0),
    ],
)
def test_scheduler_loop_waits_for_the_target_hour(monkeypatch, enabled, hour, minute, expected_sleeps, runs):
    cls = type(metaopt_settings())
    monkeypatch.setattr(cls, "META_OPT_ENABLED", property(lambda s: enabled))
    monkeypatch.setattr(cls, "META_OPT_SCHEDULE_HOUR", property(lambda s: 3))
    monkeypatch.setattr(metaopt, "datetime", SimpleNamespace(now=lambda: datetime(2026, 1, 1, hour, minute)))
    ran, sleeps = [], []
    monkeypatch.setattr(metaopt, "run_scheduled_optimization", lambda: ran.append(1))

    def sleep(seconds):
        sleeps.append(seconds)
        if seconds == 3600 and enabled or len(sleeps) == 2:
            raise _Stop

    monkeypatch.setattr(metaopt, "time", SimpleNamespace(sleep=sleep))
    with pytest.raises(_Stop):
        metaopt._scheduled_optimizer_loop()
    assert sleeps == expected_sleeps and len(ran) == runs


def test_scheduler_loop_backs_off_after_errors(monkeypatch):
    def broken(self):
        raise RuntimeError("settings down")

    monkeypatch.setattr(type(metaopt_settings()), "META_OPT_ENABLED", property(broken))
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        raise _Stop

    monkeypatch.setattr(metaopt, "time", SimpleNamespace(sleep=sleep))
    with pytest.raises(_Stop):
        metaopt._scheduled_optimizer_loop()
    assert sleeps == [600]


def test_start_scheduled_optimizer_uses_daemon_thread(monkeypatch):
    started = []

    class _Thread:
        def __init__(self, target, daemon):
            self.target, self.daemon = target, daemon

        def start(self):
            started.append(self)

    monkeypatch.setattr(metaopt.threading, "Thread", _Thread)
    t = metaopt.start_scheduled_optimizer()
    assert started == [t] and t.daemon and t.target is metaopt._scheduled_optimizer_loop


def test_main_modes(monkeypatch):
    calls = []
    monkeypatch.setattr(metaopt, "_ensure_meta_table", lambda: calls.append("ddl"))
    monkeypatch.setattr(metaopt, "run_manual_oneshot", lambda: calls.append("oneshot"))
    monkeypatch.setenv("META_OPT_MODE", " OneShot ")
    assert metaopt.main() == 0 and calls == ["ddl", "oneshot"]

    monkeypatch.setenv("META_OPT_MODE", "scheduler")
    monkeypatch.setattr(metaopt, "start_scheduled_optimizer", lambda: calls.append("thread"))
    monkeypatch.setattr(metaopt, "time", SimpleNamespace(sleep=lambda s: (_ for _ in ()).throw(_Stop())))
    with pytest.raises(_Stop):
        metaopt.main()
    assert calls[-1] == "thread"
