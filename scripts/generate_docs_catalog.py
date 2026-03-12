#!/usr/bin/env python3
# Objective: Maintenance and automation script for generate docs catalog.
"""Generate file and method documentation catalogs for the project.

Outputs:
- docs/FILE_CATALOG.md
- docs/METHOD_CATALOG.md
- docs/DOCSTRING_BACKLOG.md
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [
    ROOT / "app" / "app",
    ROOT / "app",
    ROOT / "alembic",
    ROOT / "tests",
]
DOCS_ROOT = ROOT / "docs"
FILE_OUT = DOCS_ROOT / "FILE_CATALOG.md"
METHOD_OUT = DOCS_ROOT / "METHOD_CATALOG.md"
BACKLOG_OUT = DOCS_ROOT / "DOCSTRING_BACKLOG.md"


PLACEHOLDER_PATTERNS = [
    re.compile(r"^resumo do comportamento", re.IGNORECASE),
    re.compile(r"^representa a responsabilidade", re.IGNORECASE),
    re.compile(r"^retorna o valor da configuração", re.IGNORECASE),
    re.compile(r"^valor retornado pela função", re.IGNORECASE),
]


@dataclass
class FunctionInfo:
    name: str
    line: int
    signature: str
    doc: str
    kind: str  # function|method
    class_name: Optional[str] = None


@dataclass
class ClassInfo:
    name: str
    line: int
    doc: str
    methods: List[FunctionInfo]


@dataclass
class FileInfo:
    path: Path
    module_doc: str
    classes: List[ClassInfo]
    functions: List[FunctionInfo]


def first_line(doc: Optional[str]) -> str:
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


def is_placeholder(doc: str) -> bool:
    text = (doc or "").strip()
    if not text:
        return True
    return any(p.search(text) for p in PLACEHOLDER_PATTERNS)


def build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    parts: List[str] = []
    args = node.args

    for a in args.posonlyargs:
        parts.append(a.arg)
    if args.posonlyargs:
        parts.append("/")

    for a in args.args:
        parts.append(a.arg)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for a in args.kwonlyargs:
        parts.append(a.arg)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return f"{node.name}({', '.join(parts)})"


def iter_python_files(roots: List[Path]) -> Iterable[Path]:
    seen = set()
    excluded_dir_names = {
        "__pycache__",
        ".git",
        ".pytest_cache",
        "chromadb-data",
        "ollama_data",
        "state",
        "data",
    }
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part in excluded_dir_names for part in path.parts):
                continue
            # Skip legacy compatibility modules from the primary catalogs.
            if path.name.startswith("00"):
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path


def parse_file(path: Path) -> FileInfo:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_doc = first_line(ast.get_docstring(tree) or "")

    classes: List[ClassInfo] = []
    functions: List[FunctionInfo] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(
                FunctionInfo(
                    name=node.name,
                    line=node.lineno,
                    signature=build_signature(node),
                    doc=first_line(ast.get_docstring(node) or ""),
                    kind="function",
                    class_name=None,
                )
            )
        elif isinstance(node, ast.ClassDef):
            methods: List[FunctionInfo] = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(
                        FunctionInfo(
                            name=child.name,
                            line=child.lineno,
                            signature=build_signature(child),
                            doc=first_line(ast.get_docstring(child) or ""),
                            kind="method",
                            class_name=node.name,
                        )
                    )
            classes.append(
                ClassInfo(
                    name=node.name,
                    line=node.lineno,
                    doc=first_line(ast.get_docstring(node) or ""),
                    methods=methods,
                )
            )

    return FileInfo(path=path, module_doc=module_doc, classes=classes, functions=functions)


def format_file_catalog(files: List[FileInfo]) -> str:
    lines: List[str] = []
    lines.append("# Catálogo de Arquivos")
    lines.append("")
    lines.append("Documento gerado automaticamente por `scripts/generate_docs_catalog.py`.")
    lines.append("Escopo: código Python do projeto (`app/app`, `app`, `alembic`, `tests`).")
    lines.append("")
    lines.append("| Arquivo | Módulo | Classes | Funções |")
    lines.append("|---|---|---:|---:|")
    for f in files:
        rel = f.path.relative_to(ROOT).as_posix()
        module_doc = f.module_doc or "_Sem docstring de módulo_"
        lines.append(f"| `{rel}` | {module_doc} | {len(f.classes)} | {len(f.functions)} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def format_method_catalog(files: List[FileInfo]) -> str:
    lines: List[str] = []
    lines.append("# Catálogo de Métodos e Funções")
    lines.append("")
    lines.append("Documento gerado automaticamente por `scripts/generate_docs_catalog.py`.")
    lines.append("Escopo: código Python do projeto (`app/app`, `app`, `alembic`, `tests`).")
    lines.append("")
    for f in files:
        rel = f.path.relative_to(ROOT).as_posix()
        lines.append(f"## `{rel}`")
        lines.append("")
        module_doc = f.module_doc or "_Sem docstring de módulo_"
        lines.append(f"Resumo do arquivo: {module_doc}")
        lines.append("")
        if f.functions:
            lines.append("### Funções de módulo")
            lines.append("")
            for fn in f.functions:
                desc = fn.doc or "_Sem docstring_"
                lines.append(f"- `{fn.signature}` (`{rel}:{fn.line}`): {desc}")
            lines.append("")
        if f.classes:
            lines.append("### Classes e métodos")
            lines.append("")
            for cls in f.classes:
                cdesc = cls.doc or "_Sem docstring_"
                lines.append(f"- Classe `{cls.name}` (`{rel}:{cls.line}`): {cdesc}")
                for m in cls.methods:
                    mdesc = m.doc or "_Sem docstring_"
                    lines.append(f"  - `{cls.name}.{m.signature}` (`{rel}:{m.line}`): {mdesc}")
            lines.append("")
    return "\n".join(lines) + "\n"


def format_backlog(files: List[FileInfo]) -> str:
    lines: List[str] = []
    lines.append("# Backlog de Docstrings")
    lines.append("")
    lines.append("Itens com docstring ausente ou placeholder genérico.")
    lines.append("Gerado automaticamente por `scripts/generate_docs_catalog.py`.")
    lines.append("")
    lines.append("| Item | Tipo | Local | Observação |")
    lines.append("|---|---|---|---|")

    count = 0
    for f in files:
        rel = f.path.relative_to(ROOT).as_posix()
        if is_placeholder(f.module_doc):
            obs = "docstring de módulo ausente" if not f.module_doc else "placeholder genérico"
            lines.append(f"| `{rel}` | módulo | `{rel}:1` | {obs} |")
            count += 1

        for fn in f.functions:
            if is_placeholder(fn.doc):
                obs = "docstring ausente" if not fn.doc else "placeholder genérico"
                lines.append(f"| `{fn.signature}` | função | `{rel}:{fn.line}` | {obs} |")
                count += 1

        for cls in f.classes:
            if is_placeholder(cls.doc):
                obs = "docstring ausente" if not cls.doc else "placeholder genérico"
                lines.append(f"| `{cls.name}` | classe | `{rel}:{cls.line}` | {obs} |")
                count += 1
            for m in cls.methods:
                if is_placeholder(m.doc):
                    obs = "docstring ausente" if not m.doc else "placeholder genérico"
                    lines.append(f"| `{cls.name}.{m.signature}` | método | `{rel}:{m.line}` | {obs} |")
                    count += 1

    lines.append("")
    lines.append(f"Total de itens no backlog: **{count}**.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    files = [parse_file(path) for path in iter_python_files(SCAN_ROOTS)]
    FILE_OUT.write_text(format_file_catalog(files), encoding="utf-8")
    METHOD_OUT.write_text(format_method_catalog(files), encoding="utf-8")
    BACKLOG_OUT.write_text(format_backlog(files), encoding="utf-8")
    print(f"Generated: {FILE_OUT.relative_to(ROOT)}")
    print(f"Generated: {METHOD_OUT.relative_to(ROOT)}")
    print(f"Generated: {BACKLOG_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
