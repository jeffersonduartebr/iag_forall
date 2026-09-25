# Objective: GET /research/summary: auth, research roles, tenant isolation and failure paths, on a real SQLite DB.
from __future__ import annotations

import json

import pytest
from app.api import research_routes
from app.api.auth import _encode_jwt_hs256
from app.services.pesquisa import leitura
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text  # vinculado na coleta, antes do mock global de create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

SEGREDO = "segredo-pesquisa"
_DDL = (
    """CREATE TABLE shadow_evaluations (id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, tenant TEXT,
        estrato TEXT, p_nominal REAL, executada INTEGER, motivo_corte TEXT, modelo TEXT, papel TEXT,
        regime_entrega TEXT, p_atribuicao REAL, escore_agregado REAL, painel_uniforme INTEGER,
        teria_abstido INTEGER, custo_usd REAL, latencia_s REAL, status TEXT, criado_em TEXT)""",
    """CREATE TABLE query_log (id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT, query_text TEXT, answer TEXT,
        created_at TEXT, chosen_model TEXT, estimated_cost_usd REAL, latency_s REAL, abstained INTEGER,
        judge_sampled INTEGER, decision_json TEXT)""",
)


def _sombra(conn, tenant, req, modelo, papel, escore, dia="2026-11-20"):
    conn.execute(
        text(
            "INSERT INTO shadow_evaluations (request_id, tenant, estrato, p_nominal, executada, modelo, papel, "
            "regime_entrega, escore_agregado, custo_usd, latencia_s, status, criado_em) VALUES (:r, :t, :e, 1, 1, :m, "
            ":p, 'aproveitamento', :s, 0.01, 1.5, 'ok', :c)"
        ),
        {
            "r": req,
            "t": tenant,
            "e": '{"disciplina": "bd"}',
            "m": modelo,
            "p": papel,
            "s": escore,
            "c": f"{dia} 10:00:00",
        },
    )


def _log(conn, tenant, dia, explorou):
    conn.execute(
        text(
            "INSERT INTO query_log (tenant_id, query_text, answer, created_at, chosen_model, estimated_cost_usd, "
            "latency_s, abstained, judge_sampled, decision_json) VALUES (:t, 'SEGREDO DO ALUNO', 'RESPOSTA', :c, 'X', "
            "0.2, 1.0, 0, 1, :d)"
        ),
        {"t": tenant, "c": f"{dia} 09:00:00", "d": json.dumps({"regime": {"explorou": explorou, "p_atribuicao": 0.5}})},
    )


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SEGREDO)
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with eng.begin() as conn:
        for ddl in _DDL:
            conn.execute(text(ddl))
        _sombra(conn, "ifrn-caso1", "a", "X", "entregue", 6.0)
        _sombra(conn, "ifrn-caso1", "a", "Y", "sombra", 9.0)
        _sombra(conn, "outro", "b", "Z", "entregue", 1.0)
        _sombra(conn, "outro", "b", "W", "sombra", 9.0)
        _log(conn, "ifrn-caso1", "2026-11-20", True)
        _log(conn, "ifrn-caso1", "2026-12-01", False)
        _log(conn, "outro", "2026-11-20", False)
    monkeypatch.setattr(leitura, "get_engine", lambda: eng)
    app = FastAPI()
    app.include_router(research_routes.router)
    return TestClient(app)


def _token(roles, tenant="ifrn-caso1"):
    claims = {"sub": "formativo", "roles": roles, **({"tenant_id": tenant} if tenant else {})}
    return {"Authorization": f"Bearer {_encode_jwt_hs256(claims, SEGREDO)}"}


def test_instrument_caller_gets_only_its_tenant_and_no_text(cliente):
    r = cliente.get("/research/summary", headers=_token(["instrument"]))
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["tenant"] == "ifrn-caso1"
    assert set(corpo["sombra"]["por_modelo"]) == {"X", "Y"}
    assert corpo["sombra"]["arrependimento"]["geral"]["arrependimento_medio"] == pytest.approx(3.0)
    assert corpo["regime"]["n"] == 2 and corpo["regime"]["exploracao"] == 1
    assert corpo["totais"]["n"] == 2
    assert "SEGREDO DO ALUNO" not in r.text and "RESPOSTA" not in r.text


def test_platform_admin_and_date_window(cliente):
    r = cliente.get("/research/summary?since=2026-11-01&until=2026-11-30", headers=_token(["platform_admin"]))
    assert r.status_code == 200
    assert r.json()["totais"]["n"] == 1 and r.json()["until"] == "2026-11-30"
    assert (
        cliente.get("/research/summary?since=2026-12-02&until=2026-12-01", headers=_token(["instrument"])).status_code
        == 422
    )


@pytest.mark.parametrize(
    "cabecalhos,esperado",
    [
        ({}, 401),
        ({"Authorization": "Bearer lixo"}, 401),
        (_token(["expert_reviewer"]), 403),
        (_token(["instrument"], tenant=None), 403),
    ],
)
def test_role_and_tenant_are_required(cliente, cabecalhos, esperado):
    assert cliente.get("/research/summary", headers=cabecalhos).status_code == esperado


def test_database_down_is_503(cliente, monkeypatch):
    def falha(*_a):
        raise OperationalError("SELECT", {}, Exception("fora"))

    monkeypatch.setattr(leitura, "ler", falha)
    assert cliente.get("/research/summary", headers=_token(["instrument"])).status_code == 503


def test_decoded_json_stratum_is_regrouped_as_text():
    assert leitura._normalizar({"estrato": {"b": 1, "a": 2}}) == {"estrato": '{"a": 2, "b": 1}'}
    assert leitura._normalizar({"estrato": "{}"}) == {"estrato": "{}"}
