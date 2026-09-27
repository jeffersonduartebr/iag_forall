# Objective: Shadow evaluation, research readout and regime rebuild on PostgreSQL (booleans, ON CONFLICT DO NOTHING).
"""Each test calls the real function and reads the rows back (ids in tests/pg/CATALOGO.md)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import fakeredis

RAIZ = Path(__file__).resolve().parents[2]


def _linha(**extra):
    base = {"request_id": "r1", "tenant": "t", "modelo": "m", "papel": "sombra", "status": "ok", "executada": True,
            "painel_uniforme": 1, "teria_abstido": False, "estrato": {"caso": 1}, "escores_juizes": [0.5],
            "caso": 1, "p_nominal": 0.2, "criado_em": "2026-09-20T10:00:00"}
    base.update(extra)
    return base


def test_shadow_rows_write_python_bools_and_legacy_ints_into_boolean_columns(sql):
    from app.services.sombra import registro

    assert registro.gravar([_linha(), _linha(request_id="r2", executada=0, painel_uniforme=None, teria_abstido=1)]) == 2
    linhas = sql("SELECT request_id, executada, painel_uniforme, teria_abstido, estrato FROM shadow_evaluations "
                 "ORDER BY request_id")
    assert [(r["executada"], r["painel_uniforme"], r["teria_abstido"]) for r in linhas] == [
        (True, True, False), (False, None, True)]
    assert json.loads(linhas[0]["estrato"]) == {"caso": 1}
    assert registro.gravar([]) == 0


def test_shadow_suspension_open_is_idempotent_and_close_sets_the_end_once(sql):
    from app.services.sombra import registro

    registro.abrir_suspensao("orcamento:t:20260920", "orcamento", "t", "2026-09-20T10:00:00")
    registro.abrir_suspensao("orcamento:t:20260920", "outro", "t", "2026-09-20T11:00:00")  # nada muda
    registro.fechar_suspensao("orcamento:t:20260920", "2026-09-21T00:00:05")
    registro.fechar_suspensao("orcamento:t:20260920", "2026-09-22T00:00:00")  # já fechada
    [linha] = sql("SELECT motivo, inicio, fim FROM shadow_suspensions")
    assert linha["motivo"] == "orcamento"
    assert linha["inicio"].hour == 10 and linha["fim"].day == 21


def test_research_readout_filters_tenant_and_period(sql):
    from app.services.pesquisa.leitura import ler
    from app.services.sombra import registro

    registro.gravar([_linha(), _linha(request_id="r2", tenant="outro"),
                     _linha(request_id="r3", criado_em="2026-09-25T00:00:00")])
    sql("INSERT INTO query_log (chosen_model, tenant_id, abstained, created_at) VALUES "
        "('m', 't', 1, '2026-09-20 12:00+00'), ('m', 't', 0, '2026-09-26 12:00+00'), ('m', 'x', 0, '2026-09-20 12:00+00')")
    sombra, query_log = ler("t", dt.date(2026, 9, 20), dt.date(2026, 9, 21))
    assert [s["request_id"] for s in sombra] == ["r1"] and sombra[0]["executada"] is True
    assert json.loads(sombra[0]["estrato"]) == {"caso": 1}
    assert [(q["chosen_model"], q["abstained"]) for q in query_log] == [("m", 1)]


def test_regime_window_is_rebuilt_from_decision_json(sql):
    from app.services.regime import reconstrucao

    def _decisao(p, episodio, explorou, fase="campo"):
        decisao = json.dumps({"regime": {"fase": fase, "episodio": episodio, "explorou": explorou}})
        sql("INSERT INTO query_log (chosen_model, participant, decision_json) VALUES ('m', :p, :d)", p=p, d=decisao)

    _decisao("p1", "e1", True)
    _decisao("p1", "e2", False)
    _decisao("p1", "e0", True, fase="piloto")
    _decisao("p2", "e9", True)
    assert reconstrucao._decisoes("p1") == [("e1", True), ("e2", False)]
    rds = fakeredis.FakeRedis()
    reconstrucao.garantir(rds, "p1", 10)
    assert rds.exists("regime:episodios:p1")


def test_exportar_sombra_reads_the_database_with_optional_filters(sql):
    from app.services.sombra import registro

    spec = importlib.util.spec_from_file_location("exportar_sombra", RAIZ / "scripts" / "exportar_sombra.py")
    exportar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exportar)
    registro.gravar([_linha(), _linha(request_id="r2", tenant="u", criado_em="2026-09-10T00:00:00")])
    assert [r["request_id"] for r in exportar.ler_banco(None, None)] == ["r2", "r1"]
    assert [r["request_id"] for r in exportar.ler_banco("2026-09-15", "t")] == ["r1"]
