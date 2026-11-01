# -*- coding: utf-8 -*-
# Objective: Unit tests for the LinUCB policy of the RBIE router contract.
"""Testes unitarios do LinUCB: convergencia em contexto sintetico e caracteristicas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.rbie import linucb

BRACOS = ("braco_a", "braco_b")


def _estado() -> dict:
    return {"linucb": {"alpha": 0.25, "n_atualizacoes": 0, "bracos": {}}}


def test_aprende_o_braco_certo_com_recompensas_opostas() -> None:
    """Com duas caracteristicas e recompensas opostas, o braco certo domina em poucas rodadas."""
    estado = _estado()
    contexto_a = np.array([1.0, 0.0, 1.0])
    contexto_b = np.array([0.0, 1.0, 1.0])
    escolhas = []
    for rodada in range(30):
        contexto, otimo = ((contexto_a, "braco_a") if rodada % 2 == 0 else (contexto_b, "braco_b"))
        escolhido, detalhes = linucb.escolher(estado, contexto, BRACOS)
        assert detalhes["politica"] == "linucb"
        assert detalhes["braco"] == escolhido
        assert set(detalhes["ucb"]) == set(BRACOS)
        escolhas.append(escolhido)
        recompensa = 1.0 if escolhido == otimo else 0.0
        linucb.atualizar(estado, contexto, escolhido, recompensa)

    # depois do aprendizado, as ultimas seis rodadas acertam o braco otimo
    assert escolhas[-6:] == ["braco_a", "braco_b", "braco_a", "braco_b", "braco_a", "braco_b"]
    assert estado["linucb"]["n_atualizacoes"] == 30
    # A deixou de ser identidade nos dois bracos e b deixou de ser nulo
    for braco in BRACOS:
        A = np.asarray(estado["linucb"]["bracos"][braco]["A"], dtype=float)
        b = np.asarray(estado["linucb"]["bracos"][braco]["b"], dtype=float)
        assert np.linalg.norm(A - np.eye(A.shape[0])) > 1e-6
        assert np.linalg.norm(b) > 0.0
        assert estado["linucb"]["bracos"][braco]["n"] >= 1


def test_serializacao_preserva_a_e_b() -> None:
    """A estrutura e listas JSON: ida e volta por JSON nao muda A nem b."""
    estado = _estado()
    contexto = np.array([0.3, -0.2, 1.0])
    linucb.escolher(estado, contexto, BRACOS)
    linucb.atualizar(estado, contexto, "braco_a", 0.7)
    volta = json.loads(json.dumps(estado))
    assert volta == estado


def test_caracteristicas_tem_bias_e_norma() -> None:
    """O vetor tem 64 componentes normalizados mais o bias 1."""
    caminho = Path(linucb.CAMINHO_PCA)
    if not caminho.exists():
        pytest.skip("state/rbie/pca_linucb.json ausente")
    vetor = linucb.caracteristicas("Uma redacao curta sobre educacao.", "")
    assert vetor.shape == (linucb.DIMENSAO_CARACTERISTICAS,)
    assert vetor[-1] == pytest.approx(1.0)
    assert np.linalg.norm(vetor[:-1]) == pytest.approx(1.0, abs=1e-6)


def test_prior_do_nsga_entra_em_b_e_observacoes_superam() -> None:
    """O prior do NSGA-II entra com forca n0 e as observacoes reais o superam."""
    estado = _estado()
    contexto = np.array([0.5, 0.5, 1.0])
    linucb.inicializar(estado, BRACOS, alpha=0.25, n0=20.0)
    # prior favorece braco_a (peso 1) e zera braco_b
    linucb.aplicar_prior(estado, {"braco_a": 1.0, "braco_b": 0.0})
    assert estado["linucb"]["bracos"]["braco_a"]["b_prior"] == pytest.approx(20.0)
    assert estado["linucb"]["bracos"]["braco_b"]["b_prior"] == pytest.approx(0.0)
    assert linucb.escolher(estado, contexto, BRACOS)[0] == "braco_a"
    # A continua identidade mais as observacoes; b observado nao recebe o prior
    A = np.asarray(estado["linucb"]["bracos"]["braco_a"]["A"], dtype=float)
    assert np.allclose(A, np.eye(3))
    assert np.linalg.norm(estado["linucb"]["bracos"]["braco_a"]["b"]) == 0.0
    # observacoes reais e opostas ao prior: braco_b recebe 1, braco_a recebe 0
    for _ in range(12):
        linucb.atualizar(estado, contexto, "braco_a", 0.0)
        linucb.atualizar(estado, contexto, "braco_b", 1.0)
    assert linucb.escolher(estado, contexto, BRACOS)[0] == "braco_b"
    # suavizacao: um novo ciclo nao substitui o prior aprendido, aproxima devagar
    anterior = estado["linucb"]["bracos"]["braco_a"]["b_prior"]
    linucb.aplicar_prior(estado, {"braco_a": 0.0, "braco_b": 1.0})
    novo = estado["linucb"]["bracos"]["braco_a"]["b_prior"]
    assert novo == pytest.approx((1.0 - linucb.SUAVIZACAO_PRIOR) * anterior)
