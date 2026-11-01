# -*- coding: utf-8 -*-
# Objective: Interpret the generated feedback and score agreement with human grades.
"""Interpretacao da saida do gerador e concordancia com a nota humana.

O parser replica o criterio do harness (``src/instrucoes.py::interpretar``):
JSON admitido com texto em volta, cinco competencias na ordem 1 a 5, nivel no
conjunto oficial ``{0, 40, 80, 120, 160, 200}`` e ``evidencia`` e
``proximo_passo`` nao vazios. A qualidade usada na recompensa e a concordancia
com as ``notas_humanas`` enviadas pelo harness: um menos a media das distancias
absolutas dividida pela amplitude da escala.

O escore dos juizes internos continua sendo calculado e registrado, mas nao
entra mais na recompensa.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Sequence, Tuple

NIVEIS = {0, 40, 80, 120, 160, 200}
AMPLITUDE_ESCALA = 200.0


def interpretar_saida(texto: str) -> Tuple[str, Optional[List[int]]]:
    """Devolve ``("ok", niveis)`` ou ``("formato_invalido", None)``.

    Mesmo criterio de ``instrucoes.interpretar`` do harness, sem depender do
    dicionario completo da saida.
    """
    s = re.sub(r"^```(?:json)?|```$", "", (texto or "").strip(), flags=re.M).strip()
    ini, fim = s.find("{"), s.rfind("}")
    if ini < 0 or fim <= ini:
        return "formato_invalido", None
    try:
        d = json.loads(s[ini:fim + 1])
        comps = sorted(d["competencias"], key=lambda c: int(c["competencia"]))
        niveis = [int(c["nivel"]) for c in comps]
        if [int(c["competencia"]) for c in comps] != [1, 2, 3, 4, 5]:
            return "formato_invalido", None
        if any(n not in NIVEIS for n in niveis):
            return "formato_invalido", None
        if any(not str(c.get("evidencia", "")).strip()
               or not str(c.get("proximo_passo", "")).strip() for c in comps):
            return "formato_invalido", None
        return "ok", niveis
    except (ValueError, KeyError, TypeError):
        return "formato_invalido", None


def notas_humanas_validas(notas: Any) -> Optional[List[int]]:
    """Cinco inteiros (niveis oficiais) ou ``None`` quando ausente/invalido."""
    if not isinstance(notas, (list, tuple)) or len(notas) != 5:
        return None
    valores: List[int] = []
    for nota in notas:
        if isinstance(nota, bool):
            return None
        try:
            valor = int(nota)
        except (TypeError, ValueError):
            return None
        if valor not in NIVEIS:
            return None
        valores.append(valor)
    return valores


def qualidade_concordancia(niveis_atribuidos: Optional[Sequence[int]],
                           notas_humanas: Any) -> Optional[float]:
    """Um menos a media das distancias absolutas dividida pela amplitude (0 a 1).

    Devolve ``None`` quando o parser falhou ou as notas humanas nao vieram.
    """
    humanos = notas_humanas_validas(notas_humanas)
    if humanos is None or niveis_atribuidos is None or len(niveis_atribuidos) != 5:
        return None
    distancias = [abs(int(atribuido) - int(humano))
                  for atribuido, humano in zip(niveis_atribuidos, humanos)]
    return max(0.0, min(1.0, 1.0 - (sum(distancias) / len(distancias)) / AMPLITUDE_ESCALA))
