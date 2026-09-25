# Objective: Test coverage for expert account/assessment persistence against an in-memory SQLite DB.
"""roadmap_experts: account updates and assessment listing with real SQL (SQLite)."""

import json

import pytest
from sqlalchemy import create_engine, text  # vinculado na coleta, antes do mock global de create_engine
from sqlalchemy.pool import StaticPool

from app import roadmap_experts as rx

_DDL = (
    """CREATE TABLE expert_accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE, password_hash TEXT,
        display_name TEXT, phone TEXT, enabled INTEGER, created_at TEXT, last_login_at TEXT)""",
    """CREATE TABLE expert_assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT, expert_id TEXT, benchmark_id TEXT, theme TEXT,
        query_text TEXT, answer TEXT, reference TEXT, eval_run_id TEXT, judge_quality REAL,
        quality_score REAL, rubric_json TEXT, notes TEXT, status TEXT,
        created_at TEXT, updated_at TEXT)""",
    """CREATE TABLE query_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, query_text TEXT, answer TEXT, p_entrega REAL)""",
)


@pytest.fixture
def engine(monkeypatch):
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with eng.begin() as conn:
        for ddl in _DDL:
            conn.execute(text(ddl))
    monkeypatch.setattr(rx, "get_engine", lambda: eng)
    return eng


def _assessment(conn, **fields):
    row = {"expert_id": "e1", "benchmark_id": "b", "theme": "t", "eval_run_id": "r1", "rubric_json": "{}", **fields}
    cols = ", ".join(row)
    conn.execute(text(f"INSERT INTO expert_assessments ({cols}) VALUES ({', '.join(':' + c for c in row)})"), row)


def test_update_expert_account_fields(engine):
    account_id = rx.create_expert_account(email="a@x.org", password_hash="h", display_name="Ana", phone="123")
    assert account_id > 0

    assert rx.update_expert_account(account_id) is False  # nada a atualizar
    assert rx.update_expert_account(account_id, display_name="A" * 300, phone="", enabled=False) is True
    assert rx.update_expert_account(account_id, password_hash="h2") is True
    assert rx.update_expert_account(999, enabled=True) is False  # id inexistente

    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM expert_accounts WHERE id=:i"), {"i": account_id}).mappings().one()
    assert (len(row["display_name"]), row["phone"], row["enabled"], row["password_hash"]) == (255, None, 0, "h2")


def test_list_expert_assessments_filters_and_rubric(engine):
    with engine.begin() as conn:
        _assessment(conn, rubric_json=json.dumps({"clareza": 8}))
        _assessment(conn, expert_id="e2", theme="saude", rubric_json="{quebrado")
        _assessment(conn, eval_run_id="r2", rubric_json=None)

    all_rows = rx.list_expert_assessments()
    assert [r["id"] for r in all_rows] == [3, 2, 1]  # mais recentes primeiro
    assert all_rows[2]["rubric"] == {"clareza": 8}
    assert all_rows[1]["rubric"] == {} and all_rows[0]["rubric"] == {}
    assert "rubric_json" not in all_rows[0]

    assert [r["id"] for r in rx.list_expert_assessments(expert_id="e2")] == [2]
    assert [r["id"] for r in rx.list_expert_assessments(theme="saude")] == [2]
    assert [r["id"] for r in rx.list_expert_assessments(eval_run_id="r2")] == [3]
    assert len(rx.list_expert_assessments(limit=0)) == 1  # limite mínimo 1


def test_list_expert_assessments_joins_p_entrega_from_query_log(engine):
    """p_entrega vive em query_log; sem a junção o kappa de entrega nunca tinha pares."""
    from app.services.expert_review import delivery_agreement_report

    with engine.begin() as conn:
        for q, a, p in [("q1", "r1", 0.1), ("q1", "r1", 0.9), ("q2", "r2", None), ("q3", "r3", 0.0), ("q4", "", 1.0)]:
            conn.execute(text("INSERT INTO query_log (query_text, answer, p_entrega) VALUES (:q, :a, :p)"), {"q": q, "a": a, "p": p})
        rubric = json.dumps({"scaffolding": 2})
        _assessment(conn, benchmark_id="b1", query_text="q1", answer="r1", rubric_json=rubric)  # ligada
        _assessment(conn, benchmark_id="b2", query_text="q2", answer="r2", rubric_json=rubric)  # juiz sem p_entrega
        _assessment(conn, benchmark_id="b3", query_text="q3", answer="r3", rubric_json=rubric)  # ligada, p=0.0
        _assessment(conn, benchmark_id="b4", query_text="q4", answer="", rubric_json=rubric)  # sem resposta
        _assessment(conn, benchmark_id="b5", query_text="q9", answer="r9", rubric_json=rubric)  # não chegou ao log

    by_id = {r["benchmark_id"]: r["p_entrega"] for r in rx.list_expert_assessments()}
    assert by_id == {"b1": 0.9, "b2": None, "b3": 0.0, "b4": None, "b5": None}  # a linha julgada mais recente

    report = delivery_agreement_report(rx.list_expert_assessments())
    assert report["pairs"] == 2  # só os pares ligados contam
