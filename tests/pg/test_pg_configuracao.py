# Objective: Settings, pricing seed, schema check, ROI, triggers and the correlation job on PostgreSQL.
"""Each test calls the real function and reads the rows back (ids in tests/pg/CATALOGO.md)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]


def _script(caminho: str):
    spec = importlib.util.spec_from_file_location(Path(caminho).stem, RAIZ / caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_settings_upsert_on_the_exact_key_and_reads(sql):
    import app.settings_dynamic as sd

    sd.settings.set("NSGA_W_QUALITY", "1.5", actor="teste")
    sd.settings.set("NSGA_W_QUALITY", "2.5", actor="teste")
    sd.settings.set("nsga_w_quality", "9")  # chave distinta: o PostgreSQL diferencia maiúsculas
    assert sql("SELECT setting_key, setting_value FROM settings_dynamic ORDER BY setting_key") == [
        {"setting_key": "NSGA_W_QUALITY", "setting_value": "2.5"},
        {"setting_key": "nsga_w_quality", "setting_value": "9"},
    ]
    assert sd._get_from_db("NSGA_W_QUALITY") == "2.5" and sd._get_from_db("ausente") is None
    assert sd._all_from_db() == {"NSGA_W_QUALITY": "2.5", "nsga_w_quality": "9"}
    [auditoria] = sql("SELECT count(*) AS n FROM audit_log WHERE action = 'settings.set'")
    assert auditoria["n"] == 3


def test_updated_at_trigger_moves_on_update(sql):
    sql("INSERT INTO settings_dynamic (setting_key, setting_value, updated_at) VALUES ('K', 'a', now() - interval '1 day')")
    antes = sql("SELECT updated_at FROM settings_dynamic")[0]["updated_at"]
    sql("UPDATE settings_dynamic SET setting_value = 'b'")
    depois = sql("SELECT updated_at FROM settings_dynamic")[0]["updated_at"]
    assert depois > antes


def test_pricing_seed_is_idempotent_and_is_read_by_both_readers(sql):
    from app.api.admin_models_routes import models_pricing
    from app.utils.pricing import _refresh_pricing_from_db

    from app import db_manager

    db_manager.initialize_system()
    sql("UPDATE model_pricing SET cost_input_1k = 99 WHERE model = 'gpt-4o'")
    db_manager.initialize_system()
    total = len(db_manager.PRICES) * len(db_manager.NAMESPACES)
    assert sql("SELECT count(*) AS n FROM model_pricing") == [{"n": total}]
    assert _refresh_pricing_from_db()["openai/gpt-4o"] == {"in": 0.0025, "out": 0.01}
    itens = models_pricing()["items"]
    assert len(itens) == total and itens == sorted(itens, key=lambda i: i["model"])
    assert next(i for i in itens if i["model"] == "gpt-4o")["cost_input_1k"] == 0.0025


def test_schema_check_passes_on_the_baseline_and_detects_a_missing_column(sql, pg_engine):
    from app.services.schema_check import REQUIRED_COLUMNS, missing_columns, verify_schema

    assert missing_columns(pg_engine) == []
    assert list(verify_schema(strict=True)) == []
    exigido = {**REQUIRED_COLUMNS, "query_log": (*REQUIRED_COLUMNS["query_log"], "coluna_nova"), "tabela_x": ("a",)}
    assert missing_columns(pg_engine, exigido) == ["query_log.coluna_nova", "tabela_x.a"]


def test_roi_rows_filter_by_tenant_and_window(sql):
    from app.services.roi_analytics import _load_query_rows, build_roi_report

    for tenant, idade in (("t1", "1 day"), ("t1", "40 days"), ("t2", "2 days")):
        sql("INSERT INTO query_log (chosen_model, tenant_id, quality, estimated_cost_usd, query_text, answer, created_at) "
            "VALUES ('m', :t, 8, 0.001, 'q', 'a', now() - CAST(:i AS interval))", t=tenant, i=idade)
    assert [r["tenant_id"] for r in _load_query_rows(tenant_id="t1", days=30)] == ["t1"]
    assert len(_load_query_rows(tenant_id=None, days=30)) == 2
    assert build_roi_report(tenant_id="t2", days=30)["tenant_id"] == "t2"


def test_correlation_window_and_history_append(sql, monkeypatch, pg_engine):
    import app.correlation_metrics as cm

    monkeypatch.setattr(cm, "db_engine", pg_engine)
    for i, idade in enumerate(("1 hour", "2 hours", "3 hours", "3 days")):
        sql("INSERT INTO model_metrics (model, latency_ms, cost_usd, quality_score, fitness, generation, timestamp) "
            "VALUES ('m', :l, 0.01, :q, 0.5, 1, now() - CAST(:i AS interval))", l=100.0 * (i + 1), q=5.0 + i, i=idade)
    df = cm.fetch_recent_metrics("1 day")
    assert len(df) == 3 and len(cm.fetch_recent_metrics("30 days")) == 4
    cm.persist_correlations(cm.compute_correlations(df))
    [linha] = sql("SELECT model, corr_latency_quality, generation FROM correlation_history")
    assert linha["model"] == "m" and linha["corr_latency_quality"] == pytest.approx(1.0) and linha["generation"] == 1
    cm.wait_for_db(1)


def test_calculate_savings_reads_thirty_days_without_integer_division(sql, pg_engine):
    import sys

    sys.path.insert(0, str(RAIZ / "app"))
    savings = _script("app/calculate_savings.py")
    sql("INSERT INTO query_log (chosen_model, query_text, answer, cost_per_1k, created_at) VALUES "
        "('m', 'abcde', 'xy', 0.01, now() - interval '1 day'), ('m', 'q', 'a', 0.01, now() - interval '31 days')")
    df = savings.load_query_log(pg_engine)
    assert len(df) == 1 and float(df["tokens_in_est"][0]) == 1.25 and float(df["tokens_out_est"][0]) == 0.5


def test_recalibrate_reward_gate_rows_use_a_day_window(sql):
    gate = _script("scripts/recalibrate_reward_gate.py")
    for idade in ("1 day", "10 days"):
        sql("INSERT INTO query_log (chosen_model, quality, latency_s, created_at) VALUES "
            "('m', 7, 1.0, now() - CAST(:i AS interval))", i=idade)
    sql("INSERT INTO query_log (chosen_model, quality, created_at) VALUES ('sem_latencia', 7, now())")
    assert [r["chosen_model"] for r in gate.load_rows(days=7, limit=10)] == ["m"]
    assert len(gate.load_rows(days=30, limit=10)) == 2 and len(gate.load_rows(days=30, limit=1)) == 1
