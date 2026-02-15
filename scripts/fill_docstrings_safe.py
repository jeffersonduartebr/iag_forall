#!/usr/bin/env python3
"""Safely fill missing/placeholder docstrings across project files.

Safety guarantees:
- Parses original file with AST.
- Applies edits in-memory.
- Re-parses edited content; writes only if syntactically valid.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


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
    if not line:
        return True
    return any(p.search(line) for p in PLACEHOLDER_PATTERNS)


def line_offsets(src: str) -> List[int]:
    offs = [0]
    for i, ch in enumerate(src):
        if ch == "\n":
            offs.append(i + 1)
    return offs


def idx(offs: List[int], lineno: int, col: int) -> int:
    return offs[lineno - 1] + col


def doc_node(node: ast.AST) -> Optional[ast.Expr]:
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return first
    return None


def human(name: str) -> str:
    s = name.strip("_").replace("_", " ").strip()
    return s or "processamento interno"


def module_sentence(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    return f"Módulo `{rel}`: descreve responsabilidades e integrações deste arquivo."


def class_sentence(path: Path, cls_name: str) -> str:
    return f"Classe `{cls_name}`: concentra responsabilidades de {human(path.stem)}."


def function_sentence(name: str) -> str:
    n = name.lower()
    if n in {"__init__", "__new__"}:
        return "Inicializa estado interno necessário para uso da classe."
    if n.startswith("get_"):
        return f"Obtém {human(name[4:])}."
    if n.startswith("set_"):
        return f"Define {human(name[4:])}."
    if n.startswith("list_"):
        return f"Lista {human(name[5:])}."
    if n.startswith("create_"):
        return f"Cria {human(name[7:])}."
    if n.startswith("update_"):
        return f"Atualiza {human(name[7:])}."
    if n.startswith("delete_"):
        return f"Remove {human(name[7:])}."
    if n.startswith("check_"):
        return f"Valida {human(name[6:])}."
    if n.startswith("is_"):
        return f"Indica se {human(name[3:])}."
    if n.startswith("run_"):
        return f"Executa {human(name[4:])}."
    if n.startswith("parse_"):
        return f"Interpreta {human(name[6:])}."
    if n.startswith("load_"):
        return f"Carrega {human(name[5:])}."
    if n.startswith("save_"):
        return f"Persiste {human(name[5:])}."
    if n.startswith("test_"):
        return f"Testa {human(name[5:])}."
    return f"Executa {human(name)}."


def apply_edits(src: str, edits: List[Edit]) -> str:
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


def process(path: Path) -> Tuple[bool, int]:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False, 0

    offs = line_offsets(src)
    edits: List[Edit] = []
    changed = 0

    # Module
    if is_placeholder(ast.get_docstring(tree)):
        new_module = f'"""{module_sentence(path)}"""\n\n'
        mdoc = doc_node(tree)
        if mdoc:
            s = offs[mdoc.lineno - 1]
            e = idx(offs, mdoc.end_lineno, mdoc.end_col_offset)
            if e < len(src) and src[e:e + 1] == "\n":
                e += 1
            edits.append(Edit(s, e, new_module))
        else:
            insert_at = 0
            if src.startswith("#!"):
                insert_at = src.find("\n") + 1
            if src[insert_at:insert_at + 40].startswith("# -*- coding"):
                nxt = src.find("\n", insert_at)
                insert_at = (nxt + 1) if nxt != -1 else insert_at
            edits.append(Edit(insert_at, insert_at, new_module))
        changed += 1

    queue: List[ast.AST] = list(tree.body)
    while queue:
        node = queue.pop(0)
        if isinstance(node, ast.ClassDef):
            if is_placeholder(ast.get_docstring(node)):
                dnode = doc_node(node)
                indent = " " * (node.col_offset + 4)
                repl = f'{indent}"""{class_sentence(path, node.name)}"""\n'
                if dnode:
                    s = offs[dnode.lineno - 1]
                    e = idx(offs, dnode.end_lineno, dnode.end_col_offset)
                    if e < len(src) and src[e:e + 1] == "\n":
                        e += 1
                    edits.append(Edit(s, e, repl))
                elif node.body:
                    first = node.body[0]
                    s = offs[first.lineno - 1]
                    edits.append(Edit(s, s, repl))
                changed += 1
            queue = list(node.body) + queue
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if is_placeholder(ast.get_docstring(node)):
                dnode = doc_node(node)
                indent = " " * (node.col_offset + 4)
                repl = f'{indent}"""{function_sentence(node.name)}"""\n'
                if dnode:
                    s = offs[dnode.lineno - 1]
                    e = idx(offs, dnode.end_lineno, dnode.end_col_offset)
                    if e < len(src) and src[e:e + 1] == "\n":
                        e += 1
                    edits.append(Edit(s, e, repl))
                elif node.body:
                    first = node.body[0]
                    s = offs[first.lineno - 1]
                    edits.append(Edit(s, s, repl))
                changed += 1
            queue = list(getattr(node, "body", [])) + queue

    if not edits:
        return False, 0

    candidate = apply_edits(src, edits)
    try:
        ast.parse(candidate)
    except SyntaxError:
        return False, 0

    path.write_text(candidate, encoding="utf-8")
    return True, changed


def main() -> None:
    changed_files = 0
    changed_items = 0
    for path in iter_files():
        ok, n = process(path)
        if ok:
            changed_files += 1
            changed_items += n
    print(f"Updated files: {changed_files}")
    print(f"Updated docstrings: {changed_items}")


if __name__ == "__main__":
    main()
