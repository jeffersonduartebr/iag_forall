# Objective: The bootstrap DDL and the migration chain must describe one database.
"""Two sources of truth for the schema, and nothing checking they agree.

``app.db_manager.SCHEMA_DEFINITIONS`` creates the schema on a fresh install;
the alembic chain upgrades an existing one. When they drift, a column exists on
new deployments and not on old ones, and the code that writes it fails only in
production. The history shows this has happened twice already — ``085d07f``
("esquema do db_init alinhado ao da API") and ``a372aa9`` ("a cadeia de
migrações não era executável").

``tests/test_migration_0006.py`` and ``test_migration_0007.py`` each check one
migration against the bootstrap. This checks the property that has to hold for
*all* of them at once, so a future migration cannot add a column to one side
only.
"""

import ast
import re
from pathlib import Path

import pytest

from app import db_manager

VERSIONS = Path(__file__).resolve().parents[1] / "alembic" / "versions"


def bootstrap_columns(table: str) -> set:
    """Every column the bootstrap DDL creates for one table."""
    schema = db_manager.SCHEMA_DEFINITIONS[table]
    columns = set(schema.get("columns", {}))
    for line in schema["ddl"].splitlines():
        stripped = line.strip().rstrip(",")
        if stripped and not stripped.upper().startswith(("CREATE", "UNIQUE", "INDEX", "PRIMARY", ")", "--")):
            columns.add(stripped.split()[0])
    return columns


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
# Agreement with the bootstrap
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
def test_every_migrated_column_exists_in_the_bootstrap(table):
    """Otherwise a fresh install lacks a column an upgraded one has."""
    migrated = migration_added_columns().get(table, set())
    assert migrated, f"nenhuma coluna detectada para {table}; o parser ficou obsoleto"
    missing = migrated - bootstrap_columns(table)
    assert not missing, f"{table}: migração acrescenta {sorted(missing)}, bootstrap não tem"


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
