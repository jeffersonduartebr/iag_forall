# Objective: NSGA-II, meta-optimizer, tuning and adaptive timeouts on PostgreSQL (lowercase n_pop/n_gen, windows).
"""Each test calls the real function and reads the rows back (ids in tests/pg/CATALOGO.md)."""

from __future__ import annotations

import json

import pytest


def _ema(sql, model, modality="text", lat=1.0, cost=0.01, qual=8.0, ns=""):
    sql("INSERT INTO ema_history (model, modality, ema_latency, ema_cost, ema_quality, ema_alignment, policy_namespace) "
        "VALUES (:m, :mod, :l, :c, :q, 1.0, :ns)", m=model, mod=modality, l=lat, c=cost, q=qual, ns=ns)


@pytest.fixture
def unb(monkeypatch, pg_engine):
    import app.update_nsga_best_params as modulo

    monkeypatch.setattr(modulo, "engine", pg_engine)
    monkeypatch.setattr(modulo, "rds", None)
    return modulo


def test_nsga_updater_reads_the_ema_and_upserts_weights_on_the_new_unique_key(sql, monkeypatch):
    import app.nsga_weights_updater as nwu

    monkeypatch.setattr(nwu, "redis_client", None)
    _ema(sql, "a", lat=3.0)
    dados = nwu.aggregate_ema_by_model("text", ["a", "frio"])
    assert dados["a"]["latency"] == 3.0 and dados["frio"]["latency"] == 2.0
    nwu.persist_results("text", {"a": 0.25, "b": 0.75})
    nwu.persist_results("text", {"a": 0.5})
    nwu.persist_results("vision", {"a": 1.0})
    assert sql("SELECT modality, model, weight FROM nsga_weights ORDER BY modality, model") == [
        {"modality": "text", "model": "a", "weight": 0.5},
        {"modality": "text", "model": "b", "weight": 0.75},
        {"modality": "vision", "model": "a", "weight": 1.0},
    ]


def _query_log(sql, n, quality, source, idade="0 minutes", payload=None):
    for _ in range(n):
        sql("INSERT INTO query_log (chosen_model, quality, quality_source, raw_payload, created_at) "
            "VALUES ('ollama/x', :q, :s, :p, now() - CAST(:i AS interval))", q=quality, s=source, p=payload, i=idade)


def test_judge_feedback_uses_only_the_last_hour(sql, monkeypatch):
    import app.nsga_weights_updater as nwu

    monkeypatch.setattr(nwu, "is_frozen_policy_active", lambda: False)
    chamadas = []
    monkeypatch.setattr(nwu.settings, "set", lambda chave, valor, **k: chamadas.append((chave, valor)))
    _query_log(sql, 40, 2.0, "judge", idade="2 hours")  # erros antigos: fora da janela
    _query_log(sql, 30, 9.0, "judge")
    _query_log(sql, 5, 1.0, "proxy")
    nwu.tune_weights_from_judge_feedback()
    assert chamadas == [] and nwu.JUDGE_FEEDBACK_ERROR_RATE._value.get() == 0.0
    _query_log(sql, 20, 2.0, "judge")
    nwu.tune_weights_from_judge_feedback()
    assert chamadas and chamadas[0][0] == "NSGA_W_QUALITY"
    assert nwu.JUDGE_FEEDBACK_ERROR_RATE._value.get() == pytest.approx(0.4)


def test_best_trial_uses_lowercase_columns_and_upserts_params(sql, unb):
    for trial, mean in ((1, 0.5), (2, 0.9)):
        sql("INSERT INTO nsga_meta_results (trial_id, modality, n_pop, n_gen, cxpb, mutpb, eta_c, eta_m, eff_mean, "
            "eff_std) VALUES (:t, 'text', :p, 10, 0.9, 0.1, 20, 20, :m, 0.1)", t=trial, p=8 * trial, m=mean)
    melhor = unb.load_best_trial("text")
    assert melhor["trial_id"] == 2 and melhor["n_pop"] == 16
    assert unb.load_best_trial("vision") is None
    unb.update_best_params("text", melhor)
    unb.update_best_params("text", {**melhor, "n_pop": 32})
    assert sql("SELECT id, modality, n_pop, n_gen FROM nsga_params") == [
        {"id": 1, "modality": "text", "n_pop": 32, "n_gen": 10}]


def test_model_weights_are_computed_from_the_ema_and_upserted_in_one_executemany(sql, unb):
    _ema(sql, "a", lat=1.0, cost=0.01, qual=8.0)
    _ema(sql, "b", lat=2.0, cost=0.01, qual=8.0)
    pesos = unb.compute_model_weights("text")
    assert pesos["a"] == pytest.approx(2 / 3) and unb.compute_model_weights("vision") == {}
    unb.persist_weights("text", pesos)
    unb.persist_weights("text", {"a": 0.1})
    linhas = {r["model"]: r["weight"] for r in sql("SELECT model, weight FROM nsga_weights")}
    assert linhas == {"a": 0.1, "b": pytest.approx(1 / 3)}


def test_meta_optimizer_saves_a_trial_with_lowercase_columns(sql):
    from app.nsga_meta_optimizer import save_result

    params = {"N_pop": 12, "N_gen": 5, "cxpb": 0.8, "mutpb": 0.2, "eta_c": 15, "eta_m": 25}
    save_result("vision", 7, params, 0.61, 0.02)
    assert sql("SELECT modality, trial_id, n_pop, n_gen, eff_mean FROM nsga_meta_results") == [
        {"modality": "vision", "trial_id": 7, "n_pop": 12, "n_gen": 5, "eff_mean": 0.61}]


def test_uncertainty_score_is_read_only_from_valid_json_objects(sql):
    from app.services.nsga_tuning import _recent_quality_rows

    casos = [
        json.dumps({"uncertainty_score": 0.7}),
        '{"uncertainty_score": "0.2"}',  # string numérica
        "{não é json",
        json.dumps({"uncertainty_score": None}),
        '{"uncertainty_score": NaN}',  # json.dumps(float("nan")): inválido para o PostgreSQL
        "[1, 2]",
        '{"uncertainty_score": 0.9, "t": "\\u0000"}',  # o jsonb rejeita \u0000
        json.dumps({"outra": 1}),
    ]
    for payload in casos:
        _query_log(sql, 1, 5.0, "judge", payload=payload)
    _query_log(sql, 1, 5.0, "judge", idade="25 hours", payload=json.dumps({"uncertainty_score": 0.1}))
    linhas = sorted(tuple(r) for r in _recent_quality_rows())
    assert linhas == [("ollama/x", 5.0, "0.2"), ("ollama/x", 5.0, "0.7")]


def test_adaptive_timeout_reads_the_ema_latency(sql, monkeypatch):
    import app.adaptive_timeout as at
    from app.adaptive_timeout import get_ema_latency

    monkeypatch.setattr(at, "get_redis", lambda *a, **k: None)

    _ema(sql, "m", modality="vision", lat=4.5)
    assert get_ema_latency("m", "vision") == 4.5
    assert get_ema_latency("m", "text") is None


def test_retention_deletes_only_rows_older_than_n_days(sql):
    import logging
    import threading

    from app.services.router_maintenance import retention_loop

    for idade in ("1 day", "5 days", "9 days", "30 days"):
        sql("INSERT INTO query_log (chosen_model, created_at) VALUES ('m', now() - CAST(:i AS interval))", i=idade)
    apagadas = []
    parar = threading.Event()

    def _apagou(n):
        apagadas.append(n)
        parar.set()

    from app.db import get_engine

    retention_loop(stop_event=parar, table="query_log", days=7, engine_factory=get_engine,
                   logger=logging.getLogger("t"), on_deleted=_apagou, interval=0)
    assert apagadas == [2]
    assert sql("SELECT count(*) AS n FROM query_log") == [{"n": 2}]
