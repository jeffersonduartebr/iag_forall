# -*- coding: utf-8 -*-
# Objective: Reward aligned with the human grade for the RBIE router.
"""Recompensa do roteador: concordancia com a nota humana, latencia e custo.

A qualidade deixa de vir do escore dos juizes internos (que tem correlacao
desprezivel com a nota humana) e passa a ser a concordancia entre os niveis que
o proprio roteador atribuiu e as ``notas_humanas``. A formula mantem a
combinacao da Equacao da ESPECIFICACAO: ``w_q*q + w_l*lat + w_c*cost``.

Quando a concordancia nao esta disponivel (parser invalido ou notas humanas
ausentes), a recompensa usa apenas as parcelas de custo e latencia e e marcada
como indisponivel, para nao inventar valor de qualidade.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from app.services.reward import cost_score, latency_score


def custo_por_mil(custo_usd: float, tokens_entrada: int, tokens_saida: int) -> Optional[float]:
    """Custo por mil tokens da chamada do gerador (US$)."""
    tokens = int(tokens_entrada) + int(tokens_saida)
    if tokens <= 0:
        return None
    return max(0.0, float(custo_usd)) / tokens * 1000.0


def combinar(pesos: Dict[str, Any], concordancia: Optional[float], latencia_s: float,
             custo_gerador_usd: float, tokens_entrada: int, tokens_saida: int,
             latencia_k: float = 0.12, latencia_x0: float = 20.0,
             c_base_usd_por_mil: float = 0.00096) -> Tuple[float, bool]:
    """Devolve ``(recompensa, disponivel)``.

    ``disponivel`` e falso quando a qualidade nao pode ser calculada; nesse caso
    a recompensa contem apenas as parcelas de latencia e custo.
    """
    w_q = float(pesos.get("w_q", 0.55))
    w_l = float(pesos.get("w_l", 0.30))
    w_c = float(pesos.get("w_c", 0.15))
    lat = latency_score(float(latencia_s), k=float(latencia_k), x0=float(latencia_x0))
    custo = cost_score(custo_por_mil(custo_gerador_usd, tokens_entrada, tokens_saida),
                       float(c_base_usd_por_mil))
    if concordancia is None:
        valor = w_l * lat + w_c * custo
        return max(0.0, min(1.0, float(valor))), False
    valor = w_q * max(0.0, min(1.0, float(concordancia))) + w_l * lat + w_c * custo
    return max(0.0, min(1.0, float(valor))), True
