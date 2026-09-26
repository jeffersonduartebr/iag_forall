# Objective: Every served or failed request leaves its research record: the direct query_log rows (pinned, tool turn,
# enqueue failure), request_failures, and the trace/participant/token fields of the feedback payload.
from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.services import falhas_consulta, registro_consulta
from app.services.feedback_payload import build_feedback_payload
from app.services.feedback_stages import _research_fields
from fastapi import HTTPException


def _req(**kw):
    base = dict(
        query="q",
        tenant_id="t",
        user_key="P1",
        episode_id="E1",
        pinned_model=None,
        system_prompt="gabarito",
        temperature=0.2,
        max_tokens=900,
        enable_rag_for_answer=True,
        rag_modality="text",
        rag_filter=None,
        images=None,
        image_b64=None,
        use_cache=True,
        modality="text",
    )
    return SimpleNamespace(**{**base, **kw})


def _result(**kw):
    return {
        "answer": "r",
        "model": "m",
        "modality": "text",
        "latency_s": 1.5,
        "estimated_cost_usd": 0.01,
        "finish_reason": "stop",
        "route": {"fallback": {"used": True, "models_tried": ["a", "m"]}},
        "metadata": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "reasoning_tokens": 5,
            "stage_timings_ms": {"provider_call": 900},
            "answer_before_abstention": "orig",
            "correlation_id": "c1",
        },
        **kw,
    }


def _payload(result=None):
    return build_feedback_payload(
        result or _result(),
        tenant_id="t",
        include_raw=False,
        participant="P1",
        episode_id="E1",
        route_path="/query",
        request_params=registro_consulta.parametros(_req()),
    )


def test_payload_carries_participant_tokens_and_trace():
    campos = _research_fields(_payload())
    assert campos["participant"] == "P1" and campos["episode_id"] == "E1"
    assert (campos["prompt_tokens"], campos["completion_tokens"], campos["reasoning_tokens"]) == (10, 20, 5)
    trace = campos["trace"]
    assert trace["fallback"]["models_tried"] == ["a", "m"] and trace["route_path"] == "/query"
    assert trace["stage_timings_ms"] == {"provider_call": 900} and trace["answer_before_abstention"] == "orig"
    assert trace["request"]["max_tokens"] == 900 and "gabarito" not in str(trace)  # só o hash do system prompt
    assert len(trace["request"]["system_prompt_sha256"]) == 64


@pytest.fixture
def linhas(monkeypatch):
    gravadas = []
    monkeypatch.setattr("app.query_service.insert_query_log", lambda **kw: gravadas.append(kw))
    return gravadas


@pytest.mark.parametrize(
    "req, tool, motivo",
    [
        (_req(pinned_model="x/y"), False, "pinned_model"),
        (_req(), True, "tool_turn"),
    ],
)
def test_rows_outside_the_feedback_loop_are_written_directly(monkeypatch, linhas, req, tool, motivo):
    monkeypatch.setattr("app.tasks.task_process_feedback.delay", lambda **kw: pytest.fail("não deveria enfileirar"))
    assert registro_consulta.despachar(req, _result(), None, _payload(), tool) is False
    (linha,) = linhas
    assert (linha["quality"], linha["quality_source"], linha["participant"]) == (None, motivo, "P1")
    assert linha["correlation_id"] == "c1" and linha["prompt_tokens"] == 10


def test_enqueue_failure_still_writes_the_row(monkeypatch, linhas):
    monkeypatch.setattr(
        "app.tasks.task_process_feedback.delay", lambda **kw: (_ for _ in ()).throw(RuntimeError("broker"))
    )
    assert registro_consulta.despachar(_req(), _result(), None, _payload(), False) is False
    assert [r["quality_source"] for r in linhas] == ["enqueue_failed"]


def test_ordinary_requests_go_to_the_feedback_loop(monkeypatch, linhas):
    enviados = []
    monkeypatch.setattr("app.tasks.task_process_feedback.delay", lambda **kw: enviados.append(kw))
    assert registro_consulta.despachar(_req(), _result(), None, _payload(), False) is True
    assert linhas == [] and enviados[0]["raw_payload"]["participant"] == "P1"


def test_a_failed_direct_write_never_breaks_the_response(monkeypatch):
    monkeypatch.setattr("app.query_service.insert_query_log", lambda **kw: (_ for _ in ()).throw(RuntimeError("db")))
    registro_consulta.gravar_sem_feedback(_req(), _result(), _payload(), None, "tool_turn")


def test_failures_are_classified_from_what_the_client_received():
    erro = HTTPException(504, detail={"category": "provider_timeout", "model": "m"})
    assert falhas_consulta.classificar(erro)[:3] == (504, "provider_timeout", "m")
    assert falhas_consulta.classificar(HTTPException(400, detail="bloqueado"))[:2] == (400, None)
    assert falhas_consulta.classificar(ValueError("x"))[:2] == (500, "ValueError")


class _Conn:
    def __init__(self, sql):
        self.sql = sql

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, params=None):
        self.sql.append((str(stmt), params))


def test_failed_requests_get_a_row(monkeypatch):
    sql = []
    monkeypatch.setattr("app.db.get_engine", lambda: SimpleNamespace(begin=lambda: _Conn(sql)))
    falhas_consulta.registrar(
        _req(), HTTPException(429, detail={"category": "budget"}), correlation_id="c9", route_path="/query", inicio=0.0
    )
    params = sql[-1][1]
    assert "INSERT INTO request_failures" in sql[-1][0]
    assert (params["cid"], params["status"], params["category"], params["participant"]) == ("c9", 429, "budget", "P1")


def test_recording_a_failure_never_raises(monkeypatch):
    monkeypatch.setattr("app.db.get_engine", lambda: (_ for _ in ()).throw(RuntimeError("db")))
    falhas_consulta.registrar(_req(), RuntimeError("x"), correlation_id=None, route_path=None, inicio=0.0)
