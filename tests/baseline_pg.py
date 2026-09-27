# Objective: Read the alembic_pg baseline (the schema of truth) as {table: {column: definition}} for file-only tests.
"""Parser of ``alembic_pg/versions/pg_0001_baseline.py``: tests compare code and history with it without a server."""

from __future__ import annotations

import importlib.util
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

BASELINE = Path(__file__).resolve().parents[1] / "alembic_pg" / "versions" / "pg_0001_baseline.py"
_TABELA = re.compile(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*)\)\s*$", re.S)
_NAO_COLUNA = ("PRIMARY", "CONSTRAINT", "UNIQUE", "CHECK", "FOREIGN")


@lru_cache(maxsize=1)
def ddl() -> List[str]:
    spec = importlib.util.spec_from_file_location("pg_0001_baseline", BASELINE)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return [comando.strip() for comando in modulo.DDL]


@lru_cache(maxsize=1)
def tabelas() -> Dict[str, Dict[str, str]]:
    """``{table: {column: definition}}`` of every CREATE TABLE in the baseline."""
    saida: Dict[str, Dict[str, str]] = {}
    for comando in ddl():
        casou = _TABELA.match(comando)
        if not casou:
            continue
        colunas = {}
        for linha in casou.group(2).splitlines():
            linha = linha.strip().rstrip(",")
            if linha and not linha.upper().startswith(_NAO_COLUNA):
                colunas[linha.split()[0]] = linha
        saida[casou.group(1)] = colunas
    return saida


def colunas(tabela: str) -> Dict[str, str]:
    return tabelas()[tabela]


def restricoes() -> str:
    """Every unique constraint/index statement of the baseline, joined (for substring checks)."""
    return "\n".join(ddl())
