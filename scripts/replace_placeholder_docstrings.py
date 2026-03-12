#!/usr/bin/env python3
# Objective: Maintenance and automation script for replace placeholder docstrings.
"""Replace existing placeholder docstrings safely (no insertion)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [ROOT / "app" / "app", ROOT / "app", ROOT / "alembic", ROOT / "tests"]
EXCLUDED_PARTS = {"__pycache__", ".git", ".pytest_cache", "chromadb-data", "ollama_data", "state", "data"}

PLACEHOLDER_PATTERNS = [
    re.compile(r"^resumo do comportamento", re.IGNORECASE),
    re.compile(r"^representa a responsabilidade", re.IGNORECASE),
    re.compile(r"^retorna o valor da configuração", re.IGNORECASE),
    re.compile(r"^valor retornado pela função", re.IGNORECASE),
]


@dataclass
class Edit:
    start: int
    end: int
    replacement: str


def first_line(doc: Optional[str]) -> str:
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


def is_placeholder(doc: Optional[str]) -> bool:
    line = first_line(doc)
    return bool(line and any(p.search(line) for p in PLACEHOLDER_PATTERNS))


def offsets(src: str) -> List[int]:
    out = [0]
    for i, ch in enumerate(src):
        if ch == "\n":
            out.append(i + 1)
    return out


def idx(offs: List[int], lineno: int, col: int) -> int:
    return offs[lineno - 1] + col


def doc_node(node: ast.AST):
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return first
    return None


def human(name: str) -> str:
    return (name or "").strip("_").replace("_", " ").strip() or "processamento interno"


def sentence_for(kind: str, name: str, path: Path) -> str:
    if kind == "module":
        return f"Módulo `{path.relative_to(ROOT).as_posix()}`: descreve responsabilidades e integrações deste arquivo."
    if kind == "class":
        return f"Classe `{name}`: organiza responsabilidades de {human(path.stem)}."
    lower = name.lower()
    if lower.startswith("get_"):
        return f"Obtém {human(name[4:])}."
    if lower.startswith("set_"):
        return f"Define {human(name[4:])}."
    if lower.startswith("list_"):
        return f"Lista {human(name[5:])}."
    if lower.startswith("check_"):
        return f"Valida {human(name[6:])}."
    if lower.startswith("is_"):
        return f"Indica se {human(name[3:])}."
    if lower == "__init__":
        return "Inicializa estado interno necessário para uso da classe."
    return f"Executa {human(name)}."


def apply(src: str, edits: List[Edit]) -> str:
    out = src
    for e in sorted(edits, key=lambda x: x.start, reverse=True):
        out = out[: e.start] + e.replacement + out[e.end :]
    return out


def iter_files() -> List[Path]:
    out: List[Path] = []
    seen = set()
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.py")):
            if any(part in EXCLUDED_PARTS for part in p.parts):
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out


def process(path: Path) -> int:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    offs = offsets(src)
    edits: List[Edit] = []

    mdoc = doc_node(tree)
    if mdoc and is_placeholder(ast.get_docstring(tree)):
        s = offs[mdoc.lineno - 1]
        e = idx(offs, mdoc.end_lineno, mdoc.end_col_offset)
        if e < len(src) and src[e:e + 1] == "\n":
            e += 1
        edits.append(Edit(s, e, f'"""{sentence_for("module", "", path)}"""\n'))

    queue: List[ast.AST] = list(tree.body)
    while queue:
        n = queue.pop(0)
        if isinstance(n, ast.ClassDef):
            cdoc = doc_node(n)
            if cdoc and is_placeholder(ast.get_docstring(n)):
                s = offs[cdoc.lineno - 1]
                e = idx(offs, cdoc.end_lineno, cdoc.end_col_offset)
                if e < len(src) and src[e:e + 1] == "\n":
                    e += 1
                indent = " " * (n.col_offset + 4)
                edits.append(Edit(s, e, f'{indent}"""{sentence_for("class", n.name, path)}"""\n'))
            queue = list(n.body) + queue
            continue
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fdoc = doc_node(n)
            if fdoc and is_placeholder(ast.get_docstring(n)):
                s = offs[fdoc.lineno - 1]
                e = idx(offs, fdoc.end_lineno, fdoc.end_col_offset)
                if e < len(src) and src[e:e + 1] == "\n":
                    e += 1
                indent = " " * (n.col_offset + 4)
                edits.append(Edit(s, e, f'{indent}"""{sentence_for("function", n.name, path)}"""\n'))
            queue = list(getattr(n, "body", [])) + queue

    if not edits:
        return 0
    candidate = apply(src, edits)
    try:
        ast.parse(candidate)
    except SyntaxError:
        return 0
    path.write_text(candidate, encoding="utf-8")
    return len(edits)


def main() -> None:
    total = 0
    files = 0
    for p in iter_files():
        n = process(p)
        if n:
            total += n
            files += 1
    print(f"Updated files: {files}")
    print(f"Replaced placeholder docstrings: {total}")


if __name__ == "__main__":
    main()
