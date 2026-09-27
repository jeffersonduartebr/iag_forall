# Objective: No SQL statement enters the code without a PostgreSQL test (tests/pg/CATALOGO.md).
"""Guard of the PostgreSQL verification suite.

Every SQL string literal in ``app/app/**``, ``scripts/**`` and ``app/*.py`` (a string that reads as a statement:
``SELECT ... FROM``, ``INSERT INTO``, ``UPDATE ... SET``, ``DELETE FROM``; also a ``DataFrame.to_sql`` call) belongs to a *site*: the enclosing
function, or the module-level name it is assigned to. ``tests/pg/CATALOGO.md`` has one line per statement,
``<id> | <file>::<site> | <test>``, and each named test must exist in ``tests/pg``. The suite is file-only (no
database), so it runs in the ordinary unit job too: a new statement without its line fails here, and its line
without a real PostgreSQL test fails the ``tests_postgres`` job.
"""

from __future__ import annotations

import ast
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]
CATALOGO = RAIZ / "tests" / "pg" / "CATALOGO.md"
SQL = re.compile(
    r"\bSELECT\b.+\bFROM\b|\bINSERT\s+INTO\b|\bUPDATE\s+\w+\s+(AS\s+\w+\s+)?SET\b|\bDELETE\s+FROM\b", re.S
)
#: Fora do banco da aplicação: o gabarito SQL do benchmark roda num SQLite em memória, não no PostgreSQL.
FORA_DO_BANCO = ("scripts/benchmark_v2/",)


def _arquivos(raiz: Path) -> Iterator[Path]:
    yield from sorted((raiz / "app" / "app").rglob("*.py"))
    yield from sorted((raiz / "scripts").rglob("*.py"))
    yield from sorted((raiz / "app").glob("*.py"))


def _texto(no: ast.AST) -> Optional[str]:
    if isinstance(no, ast.Constant) and isinstance(no.value, str):
        return no.value
    if isinstance(no, ast.JoinedStr):
        return "".join(v.value for v in no.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    return None


def _docstrings(arvore: ast.AST) -> set:
    ids = set()
    for no in ast.walk(arvore):
        if isinstance(no, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and no.body:
            primeiro = no.body[0]
            if isinstance(primeiro, ast.Expr) and isinstance(primeiro.value, ast.Constant):
                ids.add(id(primeiro.value))
    return ids


def _sites(arvore: ast.Module) -> Iterator[Tuple[str, ast.AST]]:
    """``(site, node)`` for every top-level function/method and module-level assignment."""
    for no in arvore.body:
        if isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield no.name, no
        elif isinstance(no, ast.ClassDef):
            for filho in no.body:
                if isinstance(filho, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{no.name}.{filho.name}", filho
        elif isinstance(no, (ast.Assign, ast.AnnAssign)):
            alvos = no.targets if isinstance(no, ast.Assign) else [no.target]
            nomes = [a.id for a in alvos if isinstance(a, ast.Name)]
            if nomes:
                yield nomes[0], no


def sql_no_codigo(raiz: Path = RAIZ) -> Dict[str, int]:
    """``{"<file>::<site>": number of SQL literals}`` over the scanned files."""
    achados: Counter = Counter()
    for caminho in _arquivos(raiz):
        relativo = caminho.relative_to(raiz).as_posix()
        if relativo.startswith(FORA_DO_BANCO):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        docs = _docstrings(arvore)
        vistos = set()
        for site, no in _sites(arvore):
            for filho in ast.walk(no):
                if isinstance(filho, ast.Call) and isinstance(filho.func, ast.Attribute) and filho.func.attr == "to_sql":
                    achados[f"{relativo}::{site}"] += 1  # DataFrame.to_sql: um INSERT sem literal
                texto = _texto(filho)
                if texto is None or id(filho) in docs or id(filho) in vistos:
                    continue
                if isinstance(filho, ast.JoinedStr):
                    vistos.update(id(v) for v in filho.values)
                if SQL.search(texto):
                    achados[f"{relativo}::{site}"] += 1
    return dict(achados)


def linhas_do_catalogo() -> List[Tuple[str, str, str]]:
    """``(id, site, test)`` of every catalog line."""
    linha = re.compile(r"^([A-Z]+\d+) \| (\S+::\S+) \| (test_\w+)$")
    return [m.groups() for m in map(linha.match, CATALOGO.read_text(encoding="utf-8").splitlines()) if m]


def _testes_pg() -> set:
    nomes = set()
    for caminho in (RAIZ / "tests" / "pg").glob("test_*.py"):
        nomes.update(re.findall(r"^def (test_\w+)", caminho.read_text(encoding="utf-8"), re.M))
    return nomes


def test_every_sql_site_is_in_the_catalog_once_per_statement():
    catalogo = Counter(site for _, site, _ in linhas_do_catalogo())
    codigo = sql_no_codigo()
    faltando = {s: n for s, n in codigo.items() if catalogo.get(s, 0) < n}
    assert not faltando, f"SQL sem linha em tests/pg/CATALOGO.md (site: statements no código): {faltando}"


def test_the_catalog_has_no_stale_lines():
    codigo = sql_no_codigo()
    sobrando = {s: n for s, n in Counter(site for _, site, _ in linhas_do_catalogo()).items() if codigo.get(s, 0) < n}
    assert not sobrando, f"linhas do catálogo sem SQL correspondente no código: {sobrando}"


def test_every_catalog_line_names_an_existing_postgresql_test():
    linhas = linhas_do_catalogo()
    ids = [i for i, _, _ in linhas]
    assert len(ids) == len(set(ids)), "ids repetidos no catálogo"
    testes = _testes_pg()
    ausentes = sorted({t for _, _, t in linhas} - testes)
    assert not ausentes, f"testes citados no catálogo que não existem em tests/pg: {ausentes}"


def test_the_scanner_sees_module_constants_functions_and_fstrings(tmp_path):
    fonte = (
        'A = text("SELECT a FROM t")\n'
        'def f():\n    """SELECT x FROM docstring is not SQL."""\n    return text(f"DELETE FROM {1} WHERE x")\n'
        'class C:\n    def g(self):\n        return "INSERT INTO t (a) VALUES (1)"\n'
    )
    (tmp_path / "app" / "app").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "app" / "app" / "m.py").write_text(fonte, encoding="utf-8")
    assert sql_no_codigo(tmp_path) == {"app/app/m.py::A": 1, "app/app/m.py::f": 1, "app/app/m.py::C.g": 1}
