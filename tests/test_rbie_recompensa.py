# -*- coding: utf-8 -*-
# Objective: Unit tests for the human-aligned reward and the output parser.
"""Testes da interpretacao da saida e da recompensa de concordancia."""

from __future__ import annotations

import pytest

from app.rbie import interpretacao, recompensa

COMPETENCIAS_OK = [
    {"competencia": 1, "nivel": 120, "evidencia": "trecho", "proximo_passo": "acao"},
    {"competencia": 2, "nivel": 80, "evidencia": "trecho", "proximo_passo": "acao"},
    {"competencia": 3, "nivel": 160, "evidencia": "trecho", "proximo_passo": "acao"},
    {"competencia": 4, "nivel": 40, "evidencia": "trecho", "proximo_passo": "acao"},
    {"competencia": 5, "nivel": 200, "evidencia": "trecho", "proximo_passo": "acao"},
]
NIVEIS = [120, 80, 160, 40, 200]


def _texto(competencias=COMPETENCIAS_OK) -> str:
    import json

    return "```json\n" + json.dumps({"competencias": competencias,
                                     "comentario_geral": "ok"}) + "\n```"


def test_parser_aceita_json_com_texto_em_volta() -> None:
    status, niveis = interpretacao.interpretar_saida(_texto())
    assert status == "ok"
    assert niveis == NIVEIS


@pytest.mark.parametrize("texto", [
    "sem json nenhum",
    '{"competencias": []}',
    '{"competencias": [{"competencia": 1, "nivel": 120, "evidencia": "", "proximo_passo": "x"}]}',
    '{"competencias": [{"competencia": 1, "nivel": 130, "evidencia": "x", "proximo_passo": "y"}]}',
    '{"competencias": [{"competencia": 1, "nivel": 120, "evidencia": "x", "proximo_passo": ""}]}',
])
def test_parser_recusa_saidas_invalidas(texto: str) -> None:
    status, niveis = interpretacao.interpretar_saida(texto)
    assert status == "formato_invalido"
    assert niveis is None


def test_concordancia_perfeita_e_maxima() -> None:
    assert interpretacao.qualidade_concordancia(NIVEIS, NIVEIS) == pytest.approx(1.0)


def test_concordancia_nula_quando_o_erro_medio_e_a_amplitude() -> None:
    # atribuidos 0 e humanos 200 nas cinco competencias: distancia media 200
    assert interpretacao.qualidade_concordancia([0, 0, 0, 0, 0], [200] * 5) == pytest.approx(0.0)


def test_concordancia_intermediaria() -> None:
    # distancia media 40 -> 1 - 40/200 = 0,8
    assert interpretacao.qualidade_concordancia([80, 80, 80, 80, 80],
                                                [120] * 5) == pytest.approx(0.8)


def test_concordancia_indisponivel_com_notas_ausentes_ou_invalidas() -> None:
    assert interpretacao.qualidade_concordancia(NIVEIS, None) is None
    assert interpretacao.qualidade_concordancia(NIVEIS, [1, 2, 3, 4, 5]) is None
    assert interpretacao.qualidade_concordancia(NIVEIS, [120] * 4) is None
    assert interpretacao.qualidade_concordancia(None, NIVEIS) is None


def test_recompensa_usa_concordancia_quando_disponivel() -> None:
    pesos = {"w_q": 0.55, "w_l": 0.30, "w_c": 0.15}
    r, disponivel = recompensa.combinar(pesos, 1.0, latencia_s=0.0, custo_gerador_usd=0.0,
                                        tokens_entrada=1000, tokens_saida=100,
                                        c_base_usd_por_mil=0.00096)
    assert disponivel is True
    assert r > 0.55  # concordancia perfeita contribui com w_q inteiro


def test_recompensa_sem_concordancia_usa_apenas_latencia_e_custo() -> None:
    pesos = {"w_q": 0.55, "w_l": 0.30, "w_c": 0.15}
    r, disponivel = recompensa.combinar(pesos, None, latencia_s=0.0, custo_gerador_usd=0.0,
                                        tokens_entrada=1000, tokens_saida=100,
                                        c_base_usd_por_mil=0.00096)
    assert disponivel is False
    # latencia zero vale ~0,9168 na logistica com k=0,12 e x0=20; custo zero vale 1
    latencia_zero = 1.0 / (1.0 + 2.718281828459045 ** (-2.4))
    esperado = 0.30 * latencia_zero + 0.15
    assert r == pytest.approx(esperado, abs=1e-6)
