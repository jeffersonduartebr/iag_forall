# Objective: Learned state on PostgreSQL: bandit posteriors, EMA, learned_state, exploration stats, centroids.
"""Each test calls the real function and reads the rows back (ids in tests/pg/CATALOGO.md)."""

from __future__ import annotations

import json


def _stats(mean, count, var=0.1, m2=0.2):
    return {"mean": mean, "count": count, "var": var, "M2": m2}


def _engine_factory():
    from app.db import get_engine

    return get_engine


def test_bandit_upsert_grows_and_never_regresses(sql):
    from app.services.bandit_stats_store import load_stats_from_db, upsert_stats_db

    upsert_stats_db(_engine_factory(), [("ctx", "m", _stats(0.5, 3))])
    upsert_stats_db(_engine_factory(), [("ctx", "m", _stats(0.6, 5, m2=0.9))])
    upsert_stats_db(_engine_factory(), [("ctx", "m", _stats(0.1, 1))])  # Redis que recomeçou do prior
    [linha] = sql("SELECT avg_reward, count, var, m2 FROM bandit_context_stats")
    assert linha == {"avg_reward": 0.6, "count": 5, "var": 0.1, "m2": 0.9}
    lido = load_stats_from_db(_engine_factory(), "ctx")
    assert lido["m"]["count"] == 5 and lido["m"]["M2"] == 0.9 and lido["m"]["mean"] == 0.6


def test_bandit_upsert_over_a_null_count_takes_the_new_posterior(sql):
    from app.services.bandit_stats_store import upsert_stats_db

    sql("INSERT INTO bandit_context_stats (context_label, model, avg_reward, count) VALUES ('c', 'm', 0.9, NULL)")
    upsert_stats_db(_engine_factory(), [("c", "m", _stats(0.3, 2))])
    [linha] = sql("SELECT avg_reward, count FROM bandit_context_stats")
    assert linha == {"avg_reward": 0.3, "count": 2}


def test_bandit_executemany_with_the_same_key_twice_in_one_batch(sql):
    from app.services.bandit_stats_store import upsert_stats_db

    upsert_stats_db(_engine_factory(), [("c", "m", _stats(0.2, 2)), ("c", "m", _stats(0.4, 4)), ("c", "x", _stats(1, 1))])
    linhas = sql("SELECT model, avg_reward, count FROM bandit_context_stats ORDER BY model")
    assert linhas == [{"model": "m", "avg_reward": 0.4, "count": 4}, {"model": "x", "avg_reward": 1.0, "count": 1}]


def test_bandit_upsert_moves_last_update(sql):
    from app.services.bandit_stats_store import upsert_stats_db

    upsert_stats_db(_engine_factory(), [("c", "m", _stats(0.2, 2))])
    sql("UPDATE bandit_context_stats SET last_update = now() - interval '1 day'")
    upsert_stats_db(_engine_factory(), [("c", "m", _stats(0.3, 3))])
    [linha] = sql("SELECT last_update > now() - interval '1 minute' AS recente FROM bandit_context_stats")
    assert linha["recente"] is True


def test_ema_upsert_on_the_four_column_key_and_log_every_tenth(sql, monkeypatch):
    from app.services import ema_persistencia as ema

    monkeypatch.setattr(ema, "_escopo", lambda: {"sem": "rubric_v1", "ns": ""})
    registro = {"ema_latency": 1.0, "ema_quality": 7.0, "ema_cost": 0.01, "updates": 9}
    assert ema.gravar("text", "m", registro)
    assert ema.gravar("text", "m", {**registro, "ema_latency": 2.0, "updates": 10})
    monkeypatch.setattr(ema, "_escopo", lambda: {"sem": "rubric_v1", "ns": "estudo"})
    assert ema.gravar("text", "m", {**registro, "updates": 1})
    linhas = sql("SELECT policy_namespace, ema_latency, updates FROM ema_history ORDER BY policy_namespace")
    assert linhas == [
        {"policy_namespace": "", "ema_latency": 2.0, "updates": 10},
        {"policy_namespace": "estudo", "ema_latency": 1.0, "updates": 1},
    ]
    assert sql("SELECT model, update_num FROM ema_history_log") == [{"model": "m", "update_num": 10}]


def test_ema_read_is_scoped_by_semantics_and_period(sql, monkeypatch):
    from app.services import ema_persistencia as ema

    monkeypatch.setattr(ema, "_escopo", lambda: {"sem": "rubric_v1", "ns": ""})
    ema.gravar("text", "a", {"ema_latency": 1.0, "ema_quality": 7.0, "ema_cost": 0.01, "updates": 3})
    ema.gravar("text", "b", {"ema_latency": 3.0, "ema_quality": 5.0, "ema_cost": 0.02, "updates": 4})
    monkeypatch.setattr(ema, "_escopo", lambda: {"sem": "formative_v1", "ns": ""})
    ema.gravar("text", "a", {"ema_latency": 9.0, "ema_quality": 1.0, "ema_cost": 0.5, "updates": 1})
    assert set(ema.carregar("text")) == {"a"} and ema.carregar("text")["a"]["ema_latency"] == 9.0
    monkeypatch.setattr(ema, "_escopo", lambda: {"sem": "rubric_v1", "ns": ""})
    assert ema.carregar("text", "b") == {
        "b": {"ema_latency": 3.0, "ema_quality": 5.0, "ema_cost": 0.02, "ema_alignment": 1.0, "updates": 4}
    }


def test_learned_state_upsert_twice_and_read(sql):
    from app.services import estado_duravel

    assert estado_duravel.gravar("centroides", {"v": 1})
    sql("UPDATE learned_state SET atualizado_em = now() - interval '1 day'")
    assert estado_duravel.gravar("centroides", {"v": 2, "texto": "ação"})
    assert estado_duravel.ler("centroides") == {"v": 2, "texto": "ação"}
    assert estado_duravel.ler("nada") is None
    [linha] = sql("SELECT count(*) AS n, bool_and(atualizado_em > now() - interval '1 minute') AS novo FROM learned_state")
    assert linha == {"n": 1, "novo": True}


def _stats_exploracao(count, reward):
    return {"count": count, "failure_count": count // 2, "mean_reward": reward, "mean_latency_s": 1.0,
            "mean_cost_usd": 0.01, "mean_observed_usd_per_1k": 0.2,
            "catalog_usd_per_1k": {"prompt_usd_per_1k": 0.1, "completion_usd_per_1k": 0.3}}


def test_exploration_stats_upsert_is_monotonic_and_keeps_the_promotion(sql):
    from app.openrouter_explorer import _persist_stats_to_db

    _persist_stats_to_db("or/m", _stats_exploracao(4, 0.5), auto_promoted=True)
    _persist_stats_to_db("or/m", _stats_exploracao(2, 0.1))  # contagem menor: não regride
    [linha] = sql("SELECT count, failure_count, mean_reward, auto_promoted_at IS NOT NULL AS promovido, stats_json "
                  "FROM openrouter_exploration_stats")
    assert (linha["count"], linha["failure_count"], linha["mean_reward"], linha["promovido"]) == (4, 2, 0.5, True)
    assert json.loads(linha["stats_json"])["count"] == 4
    _persist_stats_to_db("or/m", {**_stats_exploracao(6, 0.7), "catalog_usd_per_1k": {"prompt_usd_per_1k": 0.9}})
    [linha] = sql("SELECT count, mean_reward, catalog_prompt_usd_per_1k, auto_promoted_at IS NOT NULL AS promovido "
                  "FROM openrouter_exploration_stats")
    assert linha == {"count": 6, "mean_reward": 0.7, "catalog_prompt_usd_per_1k": 0.9, "promovido": True}


def test_exploration_stats_read_with_expanded_in(sql):
    from app.openrouter_exploration_state import _stats_do_banco
    from app.openrouter_explorer import _persist_stats_to_db

    _persist_stats_to_db("a", _stats_exploracao(3, 0.5))
    _persist_stats_to_db("b", _stats_exploracao(5, 0.6))
    _persist_stats_to_db("c", _stats_exploracao(7, 0.7))
    lido = _stats_do_banco(["a", "c", "inexistente"])
    assert set(lido) == {"a", "c"} and lido["c"]["count"] == 7
    assert _stats_do_banco([]) == {}


def test_centroid_first_id_comes_after_the_largest_cluster_label(sql):
    from app.services.centroides_duraveis import primeiro_id

    assert primeiro_id() == 0
    for rotulo in ("text|cluster:3", "vision|cluster:12", "text|default", "cluster:x"):
        sql("INSERT INTO bandit_context_stats (context_label, model) VALUES (:c, 'm')", c=rotulo)
    assert primeiro_id() == 13
