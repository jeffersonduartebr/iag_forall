# Objective: The PostgreSQL baseline, the MySQL history and the code must describe one database.
"""One owner for the schema (``alembic_pg``), and checks that everything else agrees with it.

There used to be two sources of truth — ``db_manager.SCHEMA_DEFINITIONS`` on a fresh install and the MySQL alembic
chain on an existing one — and they drifted twice (``085d07f``, ``a372aa9``). Now the runtime creates nothing: the
columns every migration of the MySQL history added must exist in the PostgreSQL baseline (or the copy loses them),
every column an ``INSERT`` of the code writes must exist there (or the insert fails on the first real request),
and ``schema_check`` must verify exactly the baseline's columns of the tables the code writes.
"""

import ast
import re
from pathlib import Path

import baseline_pg
import pytest

RAIZ = Path(__file__).resolve().parents[1]
VERSIONS = RAIZ / "alembic" / "versions"


def bootstrap_columns(table: str) -> set:
    """Every column the PostgreSQL baseline creates for one table."""
    return set(baseline_pg.colunas(table))


def migration_files():
    return sorted(p for p in VERSIONS.glob("*.py") if not p.name.startswith("__"))


# ---------------------------------------------------------------------------
# The chain itself
# ---------------------------------------------------------------------------


def test_every_migration_is_importable():
    """A file without a ``.py`` extension silently breaks the chain — and did."""
    for path in migration_files():
        ast.parse(path.read_text(encoding="utf-8"))


def test_the_versions_directory_has_no_stray_files():
    strays = [p.name for p in VERSIONS.iterdir() if p.is_file() and p.suffix not in (".py",)]
    assert strays == [], f"ficheiros sem .py não são carregados pelo alembic: {strays}"


def test_the_chain_is_linear_and_complete():
    """Two migrations claiming the same parent make the head ambiguous."""
    revisions, parents = {}, {}
    for path in migration_files():
        text = path.read_text(encoding="utf-8")
        rev = re.search(r"^revision = [\"'](.+?)[\"']", text, re.M)
        down = re.search(r"^down_revision = (.+)$", text, re.M)
        if not rev:
            continue
        revisions[rev.group(1)] = path.name
        parents[rev.group(1)] = down.group(1).strip().strip("\"'") if down else None

    assert len(set(parents.values())) == len(parents), "duas migrações com o mesmo pai"
    roots = [r for r, p in parents.items() if p in (None, "None")]
    assert len(roots) == 1, f"a cadeia tem {len(roots)} raízes: {roots}"
    for revision, parent in parents.items():
        if parent not in (None, "None"):
            assert parent in revisions, f"{revision} aponta a um pai inexistente: {parent}"


def test_the_revision_id_fits_the_version_column():
    """``alembic_version.version_num`` was VARCHAR(32) and a 33-char id failed
    mid-run, after earlier migrations had already applied."""
    for path in migration_files():
        match = re.search(r"^revision = [\"'](.+?)[\"']", path.read_text(encoding="utf-8"), re.M)
        if match:
            assert len(match.group(1)) <= 255, path.name


# ---------------------------------------------------------------------------
# Agreement with the PostgreSQL baseline
# ---------------------------------------------------------------------------


def migration_added_columns():
    """{table: {column}} declared by the migrations that expose a column map."""
    added: dict = {}
    for path in migration_files():
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"ALTER TABLE (\w+) ADD COLUMN (\w+)", text):
            added.setdefault(match.group(1), set()).add(match.group(2))
        for match in re.finditer(r"ADD COLUMN \{name\} \{definition\}", text):
            pass  # coberto pelos dicionários abaixo
        for name, const in re.findall(r"^(\w*COLUMNS)\s*=\s*\{(.*?)^\}", text, re.M | re.S):
            table = "query_log" if "QUERY_LOG" in name else "judge_logs" if "JUDGE_LOGS" in name else None
            if table:
                added.setdefault(table, set()).update(re.findall(r'^\s*"(\w+)":', const, re.M))
    return added


@pytest.mark.parametrize("table", ["query_log", "judge_logs"])
def test_every_migrated_column_exists_in_the_baseline(table):
    """Otherwise the copy from MariaDB (which has every migration applied) loses a column."""
    migrated = migration_added_columns().get(table, set())
    assert migrated, f"nenhuma coluna detectada para {table}; o parser ficou obsoleto"
    missing = {c.lower() for c in migrated} - bootstrap_columns(table)
    assert not missing, f"{table}: migração acrescenta {sorted(missing)}, o baseline PostgreSQL não tem"


def test_the_insert_only_writes_columns_that_exist():
    """A typo in the INSERT fails at runtime, on the first real request."""
    import inspect

    from app import query_service

    source = inspect.getsource(query_service.insert_query_log)
    match = re.search(r"INSERT INTO query_log\s*\((.*?)\)\s*VALUES", source, re.S)
    assert match, "não foi possível ler o INSERT"
    columns = {c.strip() for c in match.group(1).replace("\n", " ").split(",") if c.strip()}
    missing = columns - bootstrap_columns("query_log")
    assert not missing, f"o INSERT escreve colunas inexistentes: {sorted(missing)}"


def _inserts_do_codigo():
    """``(file, table, columns)`` of every ``INSERT INTO t (cols)`` literal under app/app, scripts and app/*.py."""
    padrao = re.compile(r"INSERT INTO (\w+)(?: AS \w+)?\s*\(([^)]*)\)", re.S)
    arquivos = [*(RAIZ / "app" / "app").rglob("*.py"), *(RAIZ / "scripts").glob("*.py"), *(RAIZ / "app").glob("*.py")]
    for caminho in arquivos:
        texto = caminho.read_text(encoding="utf-8")
        for tabela, colunas in padrao.findall(texto):
            nomes = {c.strip() for c in colunas.replace("\n", " ").split(",") if c.strip()}
            if all(re.fullmatch(r"[a-z_0-9]+", n) for n in nomes):
                yield caminho.relative_to(RAIZ).as_posix(), tabela, nomes


def test_every_insert_of_the_code_writes_columns_of_the_baseline():
    inserts = list(_inserts_do_codigo())
    assert len(inserts) > 20, "o parser deixou de ver os INSERTs"
    faltando = {
        (arquivo, tabela): sorted(colunas - bootstrap_columns(tabela))
        for arquivo, tabela, colunas in inserts
        if tabela not in baseline_pg.tabelas() or colunas - bootstrap_columns(tabela)
    }
    assert not faltando, f"INSERT com tabela/coluna fora do baseline alembic_pg: {faltando}"


def test_the_shadow_insert_columns_exist_too():
    """The shadow INSERT is built from ``COLUNAS``, not a literal column list."""
    from app.services.sombra.registro import COLUNAS

    assert set(COLUNAS) <= bootstrap_columns("shadow_evaluations")


def test_schema_check_verifies_exactly_the_baseline_columns_of_the_written_tables():
    from app.services.schema_check import REQUIRED_COLUMNS

    escritas = {tabela for _, tabela, _ in _inserts_do_codigo()} | {"shadow_evaluations"}
    assert escritas <= set(REQUIRED_COLUMNS), escritas - set(REQUIRED_COLUMNS)
    for tabela, colunas in REQUIRED_COLUMNS.items():
        assert list(colunas) == list(baseline_pg.colunas(tabela)), tabela


def test_the_postgresql_chain_has_one_root():
    revisoes = {}
    for caminho in (RAIZ / "alembic_pg" / "versions").glob("*.py"):
        texto = caminho.read_text(encoding="utf-8")
        rev = re.search(r"^revision = [\"'](.+?)[\"']", texto, re.M)
        down = re.search(r"^down_revision = (.+)$", texto, re.M)
        if rev:
            revisoes[rev.group(1)] = down.group(1).strip().strip("\"'") if down else None
    assert [r for r, p in revisoes.items() if p in (None, "None")] == ["pg_0001_baseline"]
