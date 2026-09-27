# Objective: What each request leaves behind on PostgreSQL: query_log, request_failures, judges and calibration.
"""Each test calls the real function and reads the rows back (ids in tests/pg/CATALOGO.md)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
from fastapi import HTTPException


def _log(**extra):
    from app.query_service import insert_query_log

    base = dict(
        query_text="pergunta\x00 com NUL", model="m", modality="text", image_provided=True, answer="resp\x00osta",
        image_output_b64=None, latency_s=1.5, estimated_cost_usd=0.002, quality=8.0, reward=0.7,
        judge_sampled=True, abstained=False, grounded=True, tenant_id="t1", correlation_id="cid-1",
        raw_payload={"uncertainty_score": 0.4, "api_key": "segredo", "nota": "a\x00b"},
        decision={"fase": "campo"}, trace={"passos": ["a\x00"]}, participant="p1", prompt_tokens=10,
        query_embedding=[0.5, 1.5], answer_embedding=np.array([2.0, -1.0]),
    )
    base.update(extra)
    insert_query_log(**base)


def test_insert_query_log_writes_every_column_strips_nul_and_keeps_embedding_bytes(sql):
    _log()
    [linha] = sql("SELECT * FROM query_log")
    assert linha["query_text"] == "pergunta com NUL" and linha["answer"] == "resposta"
    assert (linha["image_provided"], linha["judge_sampled"], linha["abstained"], linha["grounded"]) == (1, 1, 0, 1)
    assert np.frombuffer(bytes(linha["query_embedding"]), dtype=np.float32).tolist() == [0.5, 1.5]
    assert np.frombuffer(bytes(linha["answer_embedding"]), dtype=np.float32).tolist() == [2.0, -1.0]
    payload = json.loads(linha["raw_payload"])
    assert payload == {"uncertainty_score": 0.4, "api_key": "***REDACTED***", "nota": "ab"}
    assert json.loads(linha["trace_json"]) == {"passos": ["a"]}
    assert (linha["tenant_id"], linha["correlation_id"], linha["participant"], linha["prompt_tokens"]) == (
        "t1", "cid-1", "p1", 10)
    assert linha["quality_semantics"] and linha["cost_per_1k"] == 0.002 and linha["created_at"] is not None


def test_request_failure_is_recorded_with_nul_in_the_question(sql):
    from app.services.falhas_consulta import registrar

    req = SimpleNamespace(tenant_id="t", user_key="p", episode_id="e", query="oi\x00tchau", modality="text")
    registrar(req, HTTPException(429, {"category": "budget", "model": "m"}), correlation_id="c", route_path="/query",
              inicio=time.time() - 1)
    [linha] = sql("SELECT correlation_id, status_code, category, model, query_text, latency_s FROM request_failures")
    assert linha["query_text"] == "oitchau" and linha["status_code"] == 429 and linha["category"] == "budget"
    assert linha["model"] == "m" and linha["latency_s"] >= 1.0


def test_judge_performance_window_is_ten_minutes_and_is_read_back(sql):
    from app import judges

    judges._persist_judge_metrics("juiz", 0.8, 1.2, 0.001, 0.9, 0.7)
    [linha] = sql("SELECT round(extract(epoch FROM window_end - window_start)) AS s FROM judge_performance_log")
    assert linha["s"] == 600
    sql("INSERT INTO judge_performance_log (judge_model, avg_score, window_end) VALUES ('velho', 1, now() - interval '2 hours')")
    stats = judges._load_judge_stats(60)
    assert set(stats) == {"juiz"} and stats["juiz"].avg_score == 0.8 and stats["juiz"].fitness == 0.7


def test_judge_log_keeps_the_correlation_id_with_and_without_rubric(sql):
    from app.correlation import CorrelationIdContext

    from app import judges

    with CorrelationIdContext("cid-juiz"):
        judges._persist_judge_log("q\x00", "a", "juiz", 0.9, "text")
        judges._persist_judge_log("q", "a", "juiz", 0.5, "text", rubric={"clareza": 7})
    linhas = sql("SELECT query, correlation_id, rubric_json, event_type FROM judge_logs ORDER BY id")
    assert [r["correlation_id"] for r in linhas] == ["cid-juiz", "cid-juiz"]
    assert linhas[0]["query"] == "q" and linhas[0]["rubric_json"] is None
    assert json.loads(linhas[1]["rubric_json"]) == {"clareza": 7} and linhas[1]["event_type"] == "evaluation"


def test_judge_calibration_bool_and_one_hour_cache_window(sql, monkeypatch):
    from app.services import judge_calibration as jc

    monkeypatch.setenv("JUDGE_CALIBRATION_ENABLED", "1")
    jc.record_judge_calibration("juiz", "pergunta", 8.0)
    jc.record_judge_calibration("juiz", "pergunta", 6.0, was_cached=True)
    sql("UPDATE judge_calibration SET created_at = now() - interval '2 hours' WHERE predicted_score = 6.0")
    jc.update_calibration_cache_status("pergunta")
    linhas = sql("SELECT predicted_score, was_cached, cache_hit_count FROM judge_calibration ORDER BY predicted_score")
    assert linhas == [
        {"predicted_score": 6.0, "was_cached": True, "cache_hit_count": 0},  # fora da janela de 1 h
        {"predicted_score": 8.0, "was_cached": True, "cache_hit_count": 1},
    ]


def test_judge_calibration_metrics_use_the_24_hour_window(sql):
    from app.services import judge_calibration as jc

    for score, cached, idade in ((8.0, True, "1 hour"), (9.0, False, "2 hours"), (3.0, True, "30 hours")):
        sql("INSERT INTO judge_calibration (judge_model, query_hash, predicted_score, was_cached, created_at) "
            "VALUES ('juiz', 'h', :s, :c, now() - CAST(:i AS interval))", s=score, c=cached, i=idade)
    metricas = jc.get_judge_calibration_metrics()["juiz"]
    assert metricas["total_judgments"] == 2 and metricas["cached_rate"] == 0.5
    assert metricas["cache_agreement"] == 0.5 and metricas["avg_score"] == 8.5


def test_metrics_collector_sample_row(sql):
    from app.metrics_collector import _persist_sample

    _persist_sample("m", "vision", 2.0, 8.0, 0.01, cost_per_1k=0.5, tokens_in=3, vision_usage=True, generation=2)
    [linha] = sql("SELECT model, modality, latency_ms, quality_score, vision_usage, generation, fitness FROM model_metrics")
    assert linha["latency_ms"] == 2000.0 and linha["vision_usage"] == 1 and linha["generation"] == 2
    assert 0.0 < linha["fitness"] <= 1.0


def test_user_feedback_insert_and_windowed_stats(sql):
    from app.user_feedback import FeedbackType, UserFeedbackRequest, _persist_feedback, get_feedback_stats

    pedido = UserFeedbackRequest(query_id="q1", query="texto\x00", model="m", feedback_type=FeedbackType.THUMBS_UP,
                                 original_quality=6.0)
    _persist_feedback(pedido, user_quality=9.0, blended_quality=8.0, reward=0.8)
    sql("INSERT INTO user_feedback (model, feedback_type, created_at) VALUES ('m', 'thumbs_down', now() - interval '3 hours')")
    assert sql("SELECT query_text FROM user_feedback WHERE query_id = 'q1'") == [{"query_text": "texto"}]
    uma_hora = get_feedback_stats("m", hours=1)
    assert uma_hora["total_feedback"] == 1 and uma_hora["thumbs_up"] == 1 and uma_hora["thumbs_down"] == 0
    todas = get_feedback_stats(None, hours=5)
    assert todas["total_feedback"] == 2 and todas["thumbs_down"] == 1
