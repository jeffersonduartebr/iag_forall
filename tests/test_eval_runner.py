# Objective: Test coverage for evaluation-run execution (per-prompt runner and Celery task).
"""services.eval_runner and tasks.task_execute_eval_run."""

import base64
from contextlib import contextmanager

import pytest
from app.services import eval_runner as er
from fastapi import HTTPException

from app import tasks


def _resp(model="m1", quality=8.0, latency=1.5, cost=0.002, **meta):
    return {
        "result": {
            "model": model,
            "answer": "ok",
            "latency_s": latency,
            "estimated_cost_usd": cost,
            "metadata": {"quality": quality, "grounded": True, **meta},
        }
    }


class _Recorder:
    def __init__(self):
        self.rows = []

    def __call__(self, **row):
        self.rows.append(row)


def test_catalog_index_and_frozen_settings():
    meta = {
        "prompt_catalog": [{"query": " q1 ", "id": "b1"}, "lixo", {"query": "q2"}],
        "experiment_manifest": {"frozen_policy": True, "config_snapshot": {"UNCERTAINTY_THRESHOLD": 0.45}},
    }
    assert set(er.catalog_index(meta)) == {"q1", "q2"}
    assert er.frozen_settings(meta) == (True, {"UNCERTAINTY_THRESHOLD": 0.45})
    assert er.frozen_settings({"frozen_policy": 1}) == (True, None)
    assert er.frozen_settings({}) == (False, None)


def test_build_request_plain_and_multimodal(tmp_path, monkeypatch):
    opts = er.EvalRunOptions(modality="text", use_cache=True, max_tokens=64, temperature=0.1)
    run = {"policy_version": "p1", "tenant_id": "t1"}

    plain = er.build_eval_request("q", {}, run, opts)
    assert (plain.modality, plain.image_b64, plain.workload_hints) == ("text", None, None)
    assert (plain.use_cache, plain.max_tokens, plain.policy_version, plain.tenant_id) == (True, 64, "p1", "t1")

    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG")
    monkeypatch.setattr(er, "resolve_catalog_asset_path", lambda path: image)
    mm = er.build_eval_request("q", {"image_path": "img.png", "theme": "vision", "id": 7, "enable_rag": 1}, run, opts)
    assert mm.modality == "multimodal"
    assert mm.image_b64 == base64.b64encode(b"\x89PNG").decode("ascii") and mm.images == [mm.image_b64]
    assert mm.enable_rag_for_answer is True
    assert (mm.workload_hints.theme, mm.workload_hints.benchmark_id) == ("vision", "7")


def test_response_metrics_and_error_metadata():
    assert er.response_metrics({"latency_s": None, "cost_per_1k": 0.3, "metadata": {"quality": "7"}}) == (7.0, 0.0, 0.3)
    assert er.error_metadata(HTTPException(status_code=429, detail="budget")) == {"error": "budget", "status_code": 429}
    assert er.error_metadata(ValueError("boom")) == {"error": "boom"}


@pytest.mark.asyncio
async def test_run_eval_prompts_records_results_and_errors():
    answers = {"a": _resp(quality=8.0), "b": HTTPException(status_code=400, detail="blocked"), "c": RuntimeError("x")}
    seen = []

    async def process(req):
        seen.append(req.query)
        out = answers[req.query]
        if isinstance(out, Exception):
            raise out
        return out

    run = {"prompts": ["a", "b", "c"], "metadata": {"prompt_catalog": [{"query": "a", "theme": "t", "id": "x1"}]}}
    add = _Recorder()
    totals = await er.run_eval_prompts("r1", run, er.EvalRunOptions(), add, process_query=process)

    assert seen == ["a", "b", "c"]
    assert (totals.quality, totals.latency, totals.cost) == ([8.0], [1.5], [0.002])
    ok, http_err, err = add.rows
    assert ok["model"] == "m1" and ok["metadata"]["benchmark_theme"] == "t" and ok["metadata"]["grounded"] is True
    assert http_err["model"] == "error" and http_err["metadata"] == {"error": "blocked", "status_code": 400}
    assert err["metadata"] == {"error": "x"}


def test_summarize_means_over_answered_prompts():
    totals = er.EvalTotals()
    totals.add(8.0, 1.0, 0.01)
    totals.add(6.0, 3.0, 0.03)
    summary = er.summarize(3, totals)
    assert summary["n"] == 3
    assert (summary["quality_mean"], summary["latency_mean"], summary["cost_mean"]) == pytest.approx((7.0, 2.0, 0.02))
    assert er.summarize(0, er.EvalTotals())["quality_mean"] == 0.0


# ---------------------------------------------------------------- task Celery


@pytest.fixture
def task_env(monkeypatch):
    statuses, frozen_calls = [], []
    run = {"prompts": ["a"], "metadata": {"frozen_policy": True}}

    @contextmanager
    def fake_frozen(run_id, snapshot=None):
        frozen_calls.append(("enter", run_id))
        try:
            yield {}
        finally:
            frozen_calls.append(("exit", run_id))

    monkeypatch.setattr(tasks, "get_eval_run", lambda run_id: run)
    monkeypatch.setattr(
        tasks, "update_eval_run_status", lambda run_id, status, data: statuses.append((status, dict(data)))
    )
    monkeypatch.setattr("app.services.frozen_policy.frozen_policy_context", fake_frozen)
    monkeypatch.setattr("app.services.eval_feedback.apply_eval_run_feedback", lambda *a, **k: {"tuned": 1})
    return run, statuses, frozen_calls


def test_task_not_found(monkeypatch):
    monkeypatch.setattr(tasks, "get_eval_run", lambda run_id: None)
    assert tasks.task_execute_eval_run.run("missing") == {"status": "not_found", "run_id": "missing"}


def test_task_completes_under_frozen_policy(monkeypatch, task_env):
    _, statuses, frozen_calls = task_env

    async def fake_prompts(run_id, run, opts, add_result):
        assert frozen_calls == [("enter", "r1")]  # executa dentro da política congelada
        assert opts == er.EvalRunOptions(modality="text", use_cache=False, max_tokens=512, temperature=0.5)
        totals = er.EvalTotals()
        totals.add(9.0, 2.0, 0.0)
        return totals

    monkeypatch.setattr(tasks, "run_eval_prompts", fake_prompts)
    out = tasks.task_execute_eval_run.run("r1")

    assert out["status"] == "completed" and out["summary"]["quality_mean"] == 9.0
    assert out["summary"]["eval_feedback"] == {"tuned": 1}
    assert frozen_calls == [("enter", "r1"), ("exit", "r1")]
    assert [s for s, _ in statuses] == ["running", "completed", "completed"]


def test_task_failure_marks_failed_and_retries(monkeypatch, task_env):
    _, statuses, frozen_calls = task_env

    async def boom(*args):
        raise RuntimeError("provider down")

    monkeypatch.setattr(tasks, "run_eval_prompts", boom)
    monkeypatch.setattr(tasks.task_execute_eval_run, "retry", lambda exc: RuntimeError(f"retry: {exc}"))
    with pytest.raises(RuntimeError, match="retry: provider down"):
        tasks.task_execute_eval_run.run("r1")
    assert frozen_calls == [("enter", "r1"), ("exit", "r1")]
    assert statuses[-1][0] == "failed" and statuses[-1][1]["error"] == "provider down"


def test_task_feedback_failure_is_not_fatal(monkeypatch, task_env):
    run, statuses, _ = task_env
    run["metadata"] = {}

    async def fake_prompts(*args):
        return er.EvalTotals()

    def broken_feedback(*a, **k):
        raise ValueError("tuner offline")

    monkeypatch.setattr(tasks, "run_eval_prompts", fake_prompts)
    monkeypatch.setattr("app.services.eval_feedback.apply_eval_run_feedback", broken_feedback)
    out = tasks.task_execute_eval_run.run("r1")
    assert out["status"] == "completed" and "eval_feedback" not in out["summary"]
    assert [s for s, _ in statuses] == ["running", "completed"]
