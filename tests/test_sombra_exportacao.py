# Objective: The export script's regret, oracle agreement, fixed-best gain and IPW on known synthetic data (R12).
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "exportar_sombra", Path(__file__).resolve().parents[1] / "scripts" / "exportar_sombra.py"
)
exp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(exp)

S1, S2 = '{"disciplina": "bd"}', '{"disciplina": "poo"}'


def _linha(req, modelo, papel, escore=None, *, estrato=S1, regime="aproveitamento", executada=1, status="ok"):
    return {"request_id": req, "modelo": modelo, "papel": papel, "escore_agregado": escore, "estrato": estrato,
            "regime_entrega": regime, "executada": executada, "status": status, "p_nominal": 0.5,
            "motivo_corte": None if executada else status, "criado_em": f"2026-11-20T1{req[-1]}:00:00"}


LINHAS = [
    # S1: 4 sorteadas, 2 executadas -> pi = 0,5 x 2/4 = 0,25 -> peso 4
    _linha("req-1", "X", "entregue", 7.0), _linha("req-1", "Y", "sombra", 9.5),  # arrependimento 2,5
    _linha("req-2", "X", "entregue", 8.0, regime="exploracao"), _linha("req-2", "Y", "sombra", 6.0),
    _linha("req-3", "X", "entregue", executada=0, status="orcamento_esgotado"),
    _linha("req-4", "X", "entregue", executada=0, status="orcamento_esgotado"),
    # S2: 1 sorteada, 1 executada -> pi = 0,5 -> peso 2
    _linha("req-5", "Y", "entregue", 8.0, estrato=S2), _linha("req-5", "X", "sombra", 6.0, estrato=S2),
]


def test_known_regret_oracle_and_fixed_best_with_inverse_inclusion_weights():
    r = exp.analisar(LINHAS)
    assert r["pesos"] == pytest.approx({"req-1": 4.0, "req-2": 4.0, "req-5": 2.0})
    todas = r["metricas"]["todas"]["agregado"]
    assert todas["arrependimento_medio"] == pytest.approx(1.0)
    assert todas["arrependimento_acumulado"] == pytest.approx(10.0)
    assert todas["concordancia_oraculo"] == pytest.approx(0.6)
    assert todas["melhor_config_fixa"] == "Y" and todas["ganho_sobre_melhor_fixa"] == pytest.approx(-0.2)
    assert r["metricas"]["aproveitamento"]["agregado"]["n"] == 2 and r["metricas"]["exploracao"]["agregado"]["n"] == 1
    assert r["metricas"]["todas"]["por_estrato"][S2]["arrependimento_medio"] == 0.0
    assert r["contagens"] == {"sorteadas": 5, "executadas": 3, "avaliadas": 3, "cortadas_por_motivo": {"orcamento_esgotado": 2}}


def test_cli_writes_csv_and_json_with_seed_and_counts(tmp_path):
    entrada = tmp_path / "linhas.csv"
    with entrada.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(LINHAS[0]))
        escritor.writeheader()
        escritor.writerows(LINHAS)
    assert exp.main(["--entrada", str(entrada), "--saida", str(tmp_path / "out")]) == 0
    resumo = json.loads((tmp_path / "out" / "sombra_resumo.json").read_text())
    assert resumo["semente_sorteio"] == "aristo-sombra" and resumo["contagens"]["sorteadas"] == 5
    with (tmp_path / "out" / "sombra_matriz.csv").open(encoding="utf-8") as fh:
        matriz = list(csv.DictReader(fh))
    assert [m["request_id"] for m in matriz] == ["req-1", "req-2", "req-5"] and matriz[0]["Y"] == "9.5"
