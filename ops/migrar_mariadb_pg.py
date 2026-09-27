# Objective: One-off copy of the ARISTO data from MariaDB to the PostgreSQL schema (alembic_pg), with verification.
"""Cópia MariaDB -> PostgreSQL da virada para o Cloud SQL, e a conferência que decide se a virada segue.

Uso (as duas URLs no formato SQLAlchemy):
    python ops/migrar_mariadb_pg.py copiar    --origem mysql+pymysql://... --destino postgresql+psycopg2://...
    python ops/migrar_mariadb_pg.py conferir  --origem ... --destino ...

``copiar``: o esquema de destino já existe (``alembic -c alembic_pg.ini upgrade head``). Para cada tabela presente nos
dois lados: TRUNCATE no destino, cópia em lotes com nomes de coluna em minúsculas e cada valor convertido para o tipo
da coluna de destino (BOOLEAN, BYTEA, TIMESTAMPTZ em UTC, números), e as sequências IDENTITY ajustadas para max(id)+1.
``conferir``: contagem de linhas, soma/máximo de id e um hash por linha (valores normalizados) por tabela; sai com
código 1 se algo diverge. Fica fora de app/ e scripts/ de propósito: é ferramenta da virada, não código da aplicação.
"""

from __future__ import annotations

import argparse
import datetime as dt
import decimal
import hashlib
import sys
from typing import Any, Dict, List

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

LOTE = 2000
#: nsga_weights ganhou UNIQUE (modality, model) no PostgreSQL; em produção o upsert não tinha chave e duplicava.
DEDUPLICAR = {"nsga_weights": ("modality", "model")}


def _tabelas(origem: sa.Engine, destino: sa.Engine) -> List[str]:
    o = {t.lower(): t for t in sa.inspect(origem).get_table_names()}
    d = set(sa.inspect(destino).get_table_names())
    return sorted(t for t in o if t in d and t != "alembic_version")


def _converter(valor: Any, tipo: Any) -> Any:
    if valor is None:
        return None
    if isinstance(tipo, sa.Boolean):
        return bool(int(valor)) if not isinstance(valor, bool) else valor
    if isinstance(tipo, postgresql.BYTEA) or isinstance(tipo, sa.LargeBinary):
        return bytes(valor)
    if isinstance(tipo, sa.DateTime) and isinstance(valor, dt.datetime):
        return valor if valor.tzinfo else valor.replace(tzinfo=dt.timezone.utc)
    if isinstance(valor, str):
        return valor.replace("\x00", "")
    if isinstance(valor, decimal.Decimal) and isinstance(tipo, (sa.Float, sa.Integer)):
        return float(valor) if isinstance(tipo, sa.Float) else int(valor)
    return valor


def _linhas(origem: sa.Engine, tabela: str):
    with origem.connect() as conn:
        res = conn.execution_options(stream_results=True).execute(sa.text(f"SELECT * FROM `{tabela}`"))
        while lote := res.mappings().fetchmany(LOTE):
            yield [{k.lower(): v for k, v in linha.items()} for linha in lote]


def copiar(origem: sa.Engine, destino: sa.Engine) -> None:
    meta = sa.MetaData()
    meta.reflect(bind=destino)
    for nome in _tabelas(origem, destino):
        tabela = meta.tables[nome]
        tipos = {c.name: c.type for c in tabela.columns}
        vistos: set = set()
        total = 0
        with destino.begin() as conn:
            conn.execute(sa.text(f"TRUNCATE {nome} RESTART IDENTITY CASCADE"))
            for lote in _linhas(origem, nome):
                if nome in DEDUPLICAR:  # a linha de maior id vence (é a mais recente)
                    chave = DEDUPLICAR[nome]
                    lote = sorted(lote, key=lambda r: r.get("id") or 0, reverse=True)
                    lote = [r for r in lote if tuple(r[c] for c in chave) not in vistos and not vistos.add(tuple(r[c] for c in chave))]
                lote = [{c: _converter(v, tipos[c]) for c, v in r.items() if c in tipos} for r in lote]
                if lote:
                    conn.execute(tabela.insert(), lote)
                    total += len(lote)
            if "id" in tipos and tabela.c.id.identity is not None:
                conn.execute(sa.text(
                    f"SELECT setval(pg_get_serial_sequence('{nome}', 'id'), COALESCE((SELECT max(id) FROM {nome}), 0) + 1, false)"
                ))
        print(f"{nome}: {total} linha(s)")


def _normal(v: Any) -> str:
    if v is None:
        return "∅"
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (float, decimal.Decimal)):
        return f"{float(v):.6g}"
    if isinstance(v, dt.datetime):
        return (v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)).astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).hex()
    return str(v).replace("\x00", "")


def _assinatura(engine: sa.Engine, tabela: str, colunas: List[str], aspas: str) -> Dict[str, Any]:
    ordem = "id" if "id" in colunas else colunas[0]
    lista = ", ".join(f"{aspas}{c}{aspas}" for c in colunas)
    with engine.connect() as conn:
        linhas = conn.execute(sa.text(f"SELECT {lista} FROM {aspas}{tabela}{aspas} ORDER BY {aspas}{ordem}{aspas}")).all()
    h = hashlib.sha256()
    for r in linhas:
        h.update("|".join(_normal(v) for v in r).encode())
    ids = [r[colunas.index("id")] for r in linhas] if "id" in colunas else []
    return {"linhas": len(linhas), "soma_id": sum(ids), "max_id": max(ids, default=0), "hash": h.hexdigest()[:16]}


def conferir(origem: sa.Engine, destino: sa.Engine) -> bool:
    ok = True
    for nome in _tabelas(origem, destino):
        cols_o = {c["name"].lower(): c["name"] for c in sa.inspect(origem).get_columns(nome)}
        cols_d = {c["name"] for c in sa.inspect(destino).get_columns(nome)}
        comuns = sorted(c for c in cols_o if c in cols_d)
        a = _assinatura(origem, nome, [cols_o[c] for c in comuns], "`")
        b = _assinatura(destino, nome, comuns, '"')
        igual = a == b or (nome in DEDUPLICAR and b["linhas"] <= a["linhas"])
        ok &= igual
        print(f"{'OK ' if igual else 'DIVERGE'} {nome}: origem={a} destino={b}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["copiar", "conferir"])
    ap.add_argument("--origem", required=True)
    ap.add_argument("--destino", required=True)
    args = ap.parse_args()
    origem, destino = sa.create_engine(args.origem), sa.create_engine(args.destino)
    if args.acao == "copiar":
        copiar(origem, destino)
    sys.exit(0 if conferir(origem, destino) else 1)


if __name__ == "__main__":
    main()
