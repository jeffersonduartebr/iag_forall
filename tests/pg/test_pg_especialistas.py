# Objective: Expert reviewers on PostgreSQL: profiles, accounts (lower(email), RETURNING id), assessments.
"""Each test calls the real function and reads the rows back (ids in tests/pg/CATALOGO.md)."""

from __future__ import annotations

import app.roadmap_experts as rx
import pytest
from sqlalchemy.exc import IntegrityError


def test_expert_profile_upsert_keeps_fields_not_sent(sql):
    rx.upsert_expert_profile("u", display_name="Ana", theme_ids=["bd"], credentials_note="dra")
    rx.upsert_expert_profile("u", theme_ids=["bd", "so"])
    perfil = rx.get_expert_profile("u")
    assert perfil["display_name"] == "Ana" and perfil["theme_ids"] == ["bd", "so"]
    assert perfil["credentials_note"] == "dra"
    assert rx.get_expert_profile("ninguem") is None


def test_expert_account_returns_its_id_and_logs_in_case_insensitively(sql):
    primeiro = rx.create_expert_account(email="Ana@Exemplo.org", password_hash="h", display_name="Ana")
    segundo = rx.create_expert_account(email="bia@exemplo.org", password_hash="h", display_name="Bia", phone="1")
    assert (primeiro, segundo) == (1, 2)
    assert rx.get_expert_account_by_email("ana@exemplo.ORG")["id"] == primeiro
    assert rx.get_expert_account_by_email("outra@exemplo.org") is None
    with pytest.raises(IntegrityError):  # índice único em lower(email)
        rx.create_expert_account(email="ANA@exemplo.org", password_hash="h", display_name="Dup")


def test_expert_account_update_list_and_get_by_id(sql):
    conta = rx.create_expert_account(email="c@x.org", password_hash="h", display_name="C")
    sql("UPDATE expert_accounts SET updated_at = now() - interval '1 day'")
    assert rx.update_expert_account(conta, display_name="Carla", enabled=False, phone="", password_hash="h2")
    assert rx.update_expert_account(conta) is False
    lida = rx.get_expert_account_by_id(conta)
    assert (lida["display_name"], lida["enabled"], lida["phone"], lida["password_hash"]) == ("Carla", 0, None, "h2")
    assert [c["id"] for c in rx.list_expert_accounts()] == [conta]
    [linha] = sql("SELECT updated_at > now() - interval '1 minute' AS tocado FROM expert_accounts")
    assert linha["tocado"] is True


def _avaliar(**extra):
    base = dict(expert_id="e", benchmark_id="b1", theme="bd", query_text="q", answer="a", reference=None,
                eval_run_id="run", judge_quality=7.0, quality_score=8.0, rubric={"c": 1})
    base.update(extra)
    return rx.create_expert_assessment(**base)


def test_expert_assessment_upsert_returns_the_same_id_and_keeps_the_judge_quality(sql):
    primeiro = _avaliar()
    mesmo = _avaliar(quality_score=5.0, judge_quality=None, notes="revisto")
    assert primeiro == mesmo == 1
    [linha] = sql("SELECT quality_score, judge_quality, notes FROM expert_assessments")
    assert linha == {"quality_score": 5.0, "judge_quality": 7.0, "notes": "revisto"}


def test_expert_assessment_with_null_run_id_never_conflicts(sql):
    assert _avaliar(eval_run_id=None) == 1
    assert _avaliar(eval_run_id=None) == 2  # NULL não é igual a NULL numa chave única
    assert rx.list_assessed_benchmark_ids("e") == ["b1", "b1"]
    assert rx.list_assessed_benchmark_ids("e", "run") == []


def test_expert_assessment_list_carries_p_entrega_from_query_log(sql):
    _avaliar(benchmark_id="b1", query_text="q1", answer="a1")
    _avaliar(benchmark_id="b2", query_text="q2", answer="a2", judge_quality=None)
    sql("INSERT INTO query_log (query_text, answer, p_entrega) VALUES ('q1', 'a1', 0.3), ('q1', 'a1', 0.8)")
    itens = {i["benchmark_id"]: i for i in rx.list_expert_assessments(expert_id="e", theme="bd", eval_run_id="run")}
    assert itens["b1"]["p_entrega"] == 0.8 and itens["b2"]["p_entrega"] is None
    assert itens["b1"]["rubric"] == {"c": 1}
    stats = rx.get_expert_assessment_stats()
    assert (stats["total"], stats["experts"], stats["themes"], float(stats["mae"])) == (1, 1, 1, 1.0)
