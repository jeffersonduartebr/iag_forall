# Objective: Eval-run, response-review and significance helpers of roadmap_features against real SQL (SQLite).
"""roadmap_features evals: run headers/results, human review queue and per-model significance."""

import pytest
from sqlalchemy import create_engine, text  # vinculado na coleta, antes do mock global de create_engine
from sqlalchemy.pool import StaticPool

_DDL = (
    "CREATE TABLE eval_runs (id TEXT PRIMARY KEY, status TEXT, policy_version TEXT, tenant_id TEXT, notes TEXT,"
    " prompts_json TEXT, summary_json TEXT, metadata_json TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,"
    " updated_at TEXT)",
    "CREATE TABLE eval_run_results (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, prompt_text TEXT, model TEXT,"
    " quality REAL, latency_s REAL, cost_usd REAL, metadata_json TEXT, created_at TEXT)",
    "CREATE TABLE response_reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, correlation_id TEXT, tenant_id TEXT,"
    " query_text TEXT, answer TEXT, chosen_model TEXT, confidence_score REAL, confidence_band TEXT,"
    " grounded INTEGER, verification_status TEXT, review_status TEXT, review_reason TEXT, reviewer_id TEXT,"
    " reviewer_notes TEXT, corrected_answer TEXT, metadata_json TEXT, created_at TEXT, updated_at TEXT)",
)


@pytest.fixture
def rf(monkeypatch):
    from app import roadmap_features

    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with eng.begin() as conn:
        for ddl in _DDL:
            conn.execute(text(ddl))
    monkeypatch.setattr(roadmap_features, "get_engine", lambda: eng)
    monkeypatch.setattr(roadmap_features, "engine", eng, raising=False)
    return roadmap_features


def _sql(rf, sql, **params):
    with rf.engine.begin() as conn:
        conn.execute(text(sql), params)


def test_eval_run_lifecycle(rf):
    assert rf.get_eval_run("nada") is None
    rf.create_eval_run("r1", ["p1", "p2"], policy_version="v1", tenant_id="t", notes="n", metadata={"seed": 7})
    rf.add_eval_result("r1", "p1", "m1", 8.0, 1.0, 0.01, metadata={"benchmark_theme": "hist"})
    rf.add_eval_result("r1", "p2", "m1", 6.0, 3.0, 0.03)
    rf.update_eval_run_status("r1", "completed", summary={"quality_mean": 7.0})

    run = rf.get_eval_run("r1")
    assert (run["status"], run["prompts"], run["summary"], run["metadata"]) == (
        "completed",
        ["p1", "p2"],
        {"quality_mean": 7.0},
        {"seed": 7},
    )
    agg = run["aggregate"]
    assert agg["n"] == 2 and agg["quality_mean"] == pytest.approx(7.0) and agg["latency_mean"] == pytest.approx(2.0)

    results = rf.list_eval_run_results("r1")
    assert [(r["prompt_text"], r["metadata"]) for r in results] == [("p1", {"benchmark_theme": "hist"}), ("p2", {})]
    assert len(rf.list_eval_run_results("r1", limit=0)) == 1
    assert [r["id"] for r in rf.list_eval_runs()] == ["r1"]


def test_eval_run_with_corrupt_json_columns(rf):
    _sql(
        rf,
        "INSERT INTO eval_runs (id, status, prompts_json, summary_json, metadata_json) VALUES ('r2', 'x', '[', '{', '{')",
    )
    _sql(rf, "INSERT INTO eval_run_results (run_id, model, quality, metadata_json) VALUES ('r2', 'm', 1, 'nao')")
    run = rf.get_eval_run("r2")
    assert (run["prompts"], run["summary"], run["metadata"]) == ([], {}, {})
    assert run["aggregate"]["n"] == 1
    assert rf.list_eval_run_results("r2")[0]["metadata"] == {}
    assert rf.list_eval_runs()[0]["metadata"] == {}


def test_eval_runs_list_parses_metadata(rf):
    rf.create_eval_run("r3", [], metadata={"frozen_policy": True})
    listed = rf.list_eval_runs(limit=0)
    assert listed[0]["metadata"] == {"frozen_policy": True} and "metadata_json" not in listed[0]


def _review(rf, **overrides):
    fields = dict(
        correlation_id="c1",
        tenant_id="t",
        query_text="q",
        answer="a",
        chosen_model="m",
        confidence_score=0.3,
        confidence_band="low",
        grounded=True,
        verification_status="unverified",
        review_reason="low_confidence",
        metadata={"k": 1},
    )
    fields.update(overrides)
    return rf.create_response_review(**fields)


def test_response_review_queue(rf):
    first = _review(rf)
    second = _review(rf, grounded=False, metadata=None)
    assert (first, second) == (1, 2)
    queued = rf.list_response_reviews(status="needs_review")
    assert [r["id"] for r in queued] == [2, 1]
    assert queued[1]["grounded"] == 1 and queued[1]["metadata"] == {"k": 1} and queued[0]["metadata"] == {}

    assert rf.update_response_review(1, review_status="approved", reviewer_id="ana", corrected_answer="b") is True
    assert rf.update_response_review(99, review_status="approved") is False
    assert [r["id"] for r in rf.list_response_reviews(status="approved")] == [1]
    done = rf.list_response_reviews(limit=5000)
    assert len(done) == 2 and done[1]["reviewer_id"] == "ana" and done[1]["corrected_answer"] == "b"


def test_response_review_corrupt_metadata(rf):
    _sql(rf, "INSERT INTO response_reviews (review_status, metadata_json) VALUES ('needs_review', '{x')")
    assert rf.list_response_reviews()[0]["metadata"] == {}


def test_significance_report_without_results(rf):
    assert rf.eval_significance_report("vazio") == {"run_id": "vazio", "error": "no_results"}


def test_significance_report_ranks_models(rf):
    for q in (9.0, 8.5, 9.5, 8.0):
        rf.add_eval_result("r1", "p", "forte", q, 1.0, 0.02)
    for q in (4.0, 5.0, 4.5, 3.5):
        rf.add_eval_result("r1", "p", "fraco", q, 2.0, 0.01)
    _sql(rf, "INSERT INTO eval_run_results (run_id, model, quality) VALUES ('r1', NULL, NULL)")
    report = rf.eval_significance_report("r1")
    assert report["run_id"] == "r1"
    assert [m["model"] for m in report["models"]] == ["forte", "fraco", "unknown"]
    assert report["models"][0]["quality_mean"] == pytest.approx(8.75)
    assert report["models"][2]["latency_mean"] == 0.0
    fraco = next(c for c in report["comparisons"] if c["challenger_model"] == "fraco")
    assert fraco["top_model"] == "forte" and fraco["delta_quality_mean"] == pytest.approx(4.5)
