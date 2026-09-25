# Objective: The research summary's pure aggregates: per-model means, regret by day/discipline, regime and totals.
from __future__ import annotations

import json

import pytest
from app.services.pesquisa.roteamento import percentil, resumo_regime, resumo_totais
from app.services.pesquisa.sombra import disciplina, por_modelo, resumo_sombra

BD, POO = '{"disciplina": "bd"}', '{"disciplina": "poo"}'


def _linha(req, modelo, papel, escore, *, estrato=BD, dia="2026-11-20", executada=1, status="ok", **extra):
    return {
        "request_id": req,
        "modelo": modelo,
        "papel": papel,
        "escore_agregado": escore,
        "estrato": estrato,
        "regime_entrega": "aproveitamento",
        "executada": executada,
        "status": status,
        "p_nominal": 1.0,
        "motivo_corte": None if executada else status,
        "criado_em": f"{dia} 10:00:00",
        "custo_usd": 0.01,
        "latencia_s": 2.0,
        **extra,
    }


LINHAS = [
    _linha("r1", "X", "entregue", 7.0),
    _linha("r1", "Y", "sombra", 9.0, teria_abstido=1),
    _linha("r2", "X", "entregue", 8.0, dia="2026-11-21", estrato=POO),
    _linha("r2", "Y", "sombra", 6.0, dia="2026-11-21", estrato=POO),
    _linha("r3", "X", "entregue", None, executada=0, status="orcamento_esgotado"),
    _linha("r4", "Y", "sombra", None, status="erro", latencia_s=None),
]


def test_discipline_of_a_stratum_in_any_shape():
    assert disciplina(BD) == "bd" and disciplina({"disciplina": "poo"}) == "poo"
    assert disciplina("não é json") == "?" and disciplina("{}") == "?" and disciplina(None) == "?"


def test_per_model_means_cost_latency_and_roles():
    m = por_modelo(LINHAS)
    assert m["X"] == {
        "n": 2,
        "escore_medio": 7.5,
        "custo_usd": 0.02,
        "latencia_media_s": 2.0,
        "entregue": 2,
        "sombra": 0,
        "teria_abstido": 0,
    }
    assert m["Y"]["n"] == 2 and m["Y"]["sombra"] == 3 and m["Y"]["teria_abstido"] == 1
    assert m["Y"]["escore_medio"] == pytest.approx(7.5)


def test_regret_and_oracle_by_day_and_discipline():
    r = resumo_sombra(LINHAS)
    assert r["contagens"]["sorteadas"] == 4 and r["contagens"]["cortadas_por_motivo"] == {"orcamento_esgotado": 1}
    arr = r["arrependimento"]
    # r1 pesa 1/(2/3) = 1,5 (r3 sorteada e não executada no mesmo estrato x dia); r2 pesa 1
    assert arr["geral"]["n"] == 2 and arr["geral"]["arrependimento_medio"] == pytest.approx(2.0 * 1.5 / 2.5)
    assert arr["por_dia"]["2026-11-20"]["arrependimento_medio"] == pytest.approx(2.0)
    assert arr["por_dia"]["2026-11-21"]["concordancia_oraculo"] == 1.0
    assert arr["por_disciplina"]["bd"]["melhor_config_fixa"] == "Y"
    assert arr["por_disciplina"]["poo"]["melhor_config_fixa"] == "X"
    assert set(arr["por_regime"]) == {"aproveitamento"}
    assert "arrependimento_acumulado" not in arr["geral"]


def test_empty_shadow_window():
    r = resumo_sombra([])
    assert r["arrependimento"]["geral"] == {"n": 0} and r["por_modelo"] == {}


def test_percentile_interpolates():
    assert percentil([], 0.5) is None
    assert percentil([3.0], 0.9) == 3.0
    assert percentil([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert percentil([4.0, 1.0, 3.0, 2.0], 0.9) == pytest.approx(3.7)


def _q(dia, regime=None, **extra):
    decisao = json.dumps({"regime": regime}) if regime is not None else extra.pop("decision_json", None)
    return {
        "created_at": f"{dia} 09:00:00",
        "chosen_model": "X",
        "estimated_cost_usd": 0.5,
        "latency_s": 1.0,
        "abstained": 0,
        "judge_sampled": 1,
        "decision_json": decisao,
        **extra,
    }


def test_regime_counts_reasons_regenerations_and_series():
    linhas = [
        _q("2026-11-20", {"explorou": True, "p_atribuicao": 0.05, "regenerado": True}),
        _q("2026-11-20", {"explorou": False, "p_atribuicao": 0.85}),
        _q("2026-11-21", {"explorou": False, "p_atribuicao": 1.0, "motivo_sem_exploracao": "teto_janela"}),
        _q("2026-11-21", decision_json={"regime": {"explorou": False, "motivo_sem_exploracao": "politica_congelada"}}),
        _q("2026-11-21", decision_json="{quebrado"),
        _q("2026-11-21", decision_json="[1]"),
        _q("2026-11-21"),
    ]
    r = resumo_regime(linhas)
    assert (r["n"], r["exploracao"], r["aproveitamento"], r["regenerados"]) == (4, 1, 3, 1)
    assert r["motivos_sem_exploracao"] == {"teto_janela": 1, "politica_congelada": 1}
    assert r["p_atribuicao_media"] == pytest.approx((0.05 + 0.85 + 1.0) / 3)
    assert r["por_dia"] == {"2026-11-20": {"exploracao": 1, "aproveitamento": 1}, "2026-11-21": {"aproveitamento": 2}}
    assert resumo_regime([])["p_atribuicao_media"] is None


def test_totals_cost_per_day_latency_and_rates():
    linhas = [
        _q("2026-11-20"),
        _q("2026-11-20", chosen_model="Y", latency_s=3.0, abstained=1),
        _q("2026-11-21", estimated_cost_usd=None, latency_s=None, judge_sampled=0),
    ]
    t = resumo_totais(linhas)
    assert t["custo_por_dia_modelo"] == {"2026-11-20": {"X": 0.5, "Y": 0.5}, "2026-11-21": {"X": 0.0}}
    assert t["custo_total_usd"] == 1.0 and t["latencia_p50_s"] == 2.0
    assert t["taxa_abstencao"] == pytest.approx(1 / 3) and t["taxa_juiz_amostrado"] == pytest.approx(2 / 3)
    vazio = resumo_totais([])
    assert vazio["taxa_abstencao"] is None and vazio["latencia_p90_s"] is None
