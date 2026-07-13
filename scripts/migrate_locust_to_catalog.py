#!/usr/bin/env python3
"""Extract queries from locustfile.py into benchmark JSONL catalog."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCUST_PATH = ROOT / "tests" / "locustfile.py"
MANIFEST_PATH = ROOT / "data" / "benchmark_queries" / "manifest.yaml"
CATALOG_DIR = ROOT / "data" / "benchmark_queries"

DUP_MARKER = "What were the primary motivations for British colonization of North America?"
ORPHAN_MATH_START = "What is the formal epsilon-delta definition of a limit?"

# Theme extraction order: (theme_id, start_header_substring, extra_sections)
THEME_SECTIONS: List[Tuple[str, str, List[str]]] = [
    ("historia", "🏛️ História", []),
    ("programacao", "💻 Programação", []),
    ("fisica", "⚛️ Física", []),
    ("quimica", "🧪 Química", []),
    ("matematica", "📊 Matemática", ["__orphan_math__"]),
    ("agro_iot", "🌾 Agronegócio", []),
    ("geografia", "🌎 Geografia (45)", []),
    ("redes", "📡 Redes", []),
    ("ia", "⚙️ Inteligência Artificial", []),
    ("biologia", "🧬 Biologia", []),
    ("economia", "📈 Economia", []),
    ("literatura", "📚 Literatura", []),
    ("filosofia", "🧠 Filosofia", []),
    ("arte", "🎨 Arte", []),
    ("direito", "⚖️ Direito", []),
    ("cinema", "🎬 Cinema", []),
    ("culinaria", "🍳 Culinária", []),
    ("astronomia", "🌌 Astronomia", []),
    ("transversais", "Questões Transversais (Complexas)", []),
    ("foco_meta", "Questões Transversais (Foco Meta)", []),
    ("brasil_historia", "História do Brasil", []),
    ("brasil_geografia", "Geografia e Geopolítica do Brasil", []),
    ("brasil_economia", "Economia Brasileira", []),
    ("brasil_politica", "Política Brasileira", []),
    ("brasil_cultura", "Cultura e Sociedade Brasileira", []),
    ("america_norte", "Perguntas Gerais sobre a américa do norte", ["25 questões muito difíceis sobre américa do norte"]),
    ("europa", "100 questões sobre a Europa", []),
    ("matematica_avancada", "25 questões avançadas sobre matemática", []),
    ("asia", "100 questões sobre a Ásia", ["25 questões difíceis sobre a Ásia"]),
    ("medicina", "100 questões sobre medicina", ["25 questões difíceis sobre medicina"]),
    ("geografia_regional", "100 questões sobre geografia", ["25 questões difíceis sobre geografia"]),
]

HARD_MARKERS = (
    "25 questões muito difíceis",
    "25 questões difíceis",
    "25 questões avançadas",
)
COMPLEX_MARKER = "# Complexas (5)"


def _load_queries_block(text: str) -> str:
    start = text.index("QUERIES = [")
    dup_idx = text.find(DUP_MARKER)
    dup_idx2 = text.find(DUP_MARKER, dup_idx + 1) if dup_idx >= 0 else -1
    if dup_idx2 > 0:
        # truncate before duplicate tail inside array
        chunk_start = start
        # find line start of second dup query
        line_start = text.rfind("\n", 0, dup_idx2)
        end = text.find("]", dup_idx2)
        # use content only up to line before duplicate block
        pre = text[chunk_start:line_start]
        # but we need closing - parse ast on truncated version
        truncated = pre.rstrip() + "\n]\n"
        return truncated.split("QUERIES = ", 1)[1]
    end = text.find("\n]", start)
    return text[start + len("QUERIES = ") : end + 1]


def _parse_query_lines(block: str) -> List[Tuple[str, str]]:
    """Return list of (query_text, context_hint)."""
    lines = block.splitlines()
    results: List[Tuple[str, str]] = []
    ctx = "medium"
    for line in lines:
        stripped = line.strip()
        if COMPLEX_MARKER in stripped:
            ctx = "complex"
            continue
        if any(m in stripped for m in HARD_MARKERS):
            ctx = "hard"
            continue
        if stripped.startswith("# Originais"):
            ctx = "easy"
            continue
        if stripped.startswith("# Novas"):
            ctx = "medium"
            continue
        if stripped.startswith("#") or not stripped:
            continue
        if stripped.startswith('{"query":'):
            try:
                obj = ast.literal_eval(stripped.rstrip(","))
                q = str(obj.get("query", "")).strip()
                if q:
                    results.append((q, ctx))
            except (SyntaxError, ValueError):
                continue
    return results


def _section_slice(lines: List[str], start_pat: str, end_patterns: List[str]) -> List[str]:
    start_idx = next(i for i, ln in enumerate(lines) if start_pat in ln)
    end_idx = len(lines)
    for pat in end_patterns:
        for i in range(start_idx + 1, len(lines)):
            if pat in lines[i] and lines[i].strip().startswith("#"):
                end_idx = min(end_idx, i)
                break
    return lines[start_idx:end_idx]


def _extract_orphan_math(lines: List[str]) -> List[str]:
    out: List[str] = []
    in_orphan = False
    for ln in lines:
        if ORPHAN_MATH_START in ln:
            in_orphan = True
        if in_orphan:
            if "25 questões avançadas sobre matemática" in ln:
                break
            out.append(ln)
    return out


def _infer_lang(theme_id: str, query: str) -> str:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    theme = next(t for t in manifest["themes"] if t["id"] == theme_id)
    default = str(theme.get("default_lang", "pt"))
    if default != "mixed":
        return default
    ascii_ratio = sum(1 for c in query if c.isascii()) / max(len(query), 1)
    return "en" if ascii_ratio > 0.85 else "pt"


def _write_theme(theme_id: str, queries: List[Tuple[str, str]]) -> int:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    theme = next(t for t in manifest["themes"] if t["id"] == theme_id)
    path = CATALOG_DIR / str(theme["file"])
    seen: set[str] = set()
    rows: List[Dict[str, Any]] = []
    for q, difficulty in queries:
        norm = re.sub(r"\s+", " ", q.strip().lower())
        if norm in seen:
            continue
        seen.add(norm)
        seq = len(rows) + 1
        rows.append(
            {
                "id": f"{theme_id}_{seq:03d}",
                "query": q,
                "theme": theme_id,
                "difficulty": difficulty,
                "lang": _infer_lang(theme_id, q),
                "tags": [],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    text = LOCUST_PATH.read_text(encoding="utf-8")
    block = _load_queries_block(text)
    lines = block.splitlines()

    # Build end markers from theme order
    headers = [t[1] for t in THEME_SECTIONS]
    for idx, (theme_id, start_pat, extras) in enumerate(THEME_SECTIONS):
        end_pats: List[str] = []
        if idx + 1 < len(THEME_SECTIONS):
            end_pats.append(THEME_SECTIONS[idx + 1][1])
        for ex in extras:
            if ex != "__orphan_math__":
                end_pats.append(ex)

        if theme_id == "europa":
            # Stop before orphan math block
            section_lines = []
            start_i = next(i for i, ln in enumerate(lines) if start_pat in ln)
            for i in range(start_i, len(lines)):
                if ORPHAN_MATH_START in lines[i]:
                    break
                if "25 questões avançadas sobre matemática" in lines[i]:
                    break
                if i > start_i and THEME_SECTIONS[idx + 1][1] in lines[i]:
                    break
                section_lines.append(lines[i])
        elif theme_id == "matematica":
            main_sec = _section_slice(lines, start_pat, [THEME_SECTIONS[idx + 1][1]])
            orphan_sec = _extract_orphan_math(lines)
            section_lines = main_sec + orphan_sec
        elif extras:
            section_lines = []
            start_i = next(i for i, ln in enumerate(lines) if start_pat in ln)
            end_i = len(lines)
            for pat in end_pats:
                for j in range(start_i + 1, len(lines)):
                    if pat in lines[j] and lines[j].strip().startswith("#"):
                        end_i = min(end_i, j)
            section_lines = lines[start_i:end_i]
            for ex in extras:
                if ex == "__orphan_math__":
                    continue
                ex_start = next((j for j, ln in enumerate(lines) if ex in ln), None)
                if ex_start is not None:
                    ex_end = len(lines)
                    for pat in end_pats:
                        for j in range(ex_start + 1, len(lines)):
                            if pat in lines[j] and lines[j].strip().startswith("#"):
                                ex_end = min(ex_end, j)
                    section_lines.extend(lines[ex_start:ex_end])
        else:
            section_lines = _section_slice(lines, start_pat, end_pats)

        parsed = _parse_query_lines("\n".join(section_lines))
        count = _write_theme(theme_id, parsed)
        print(f"{theme_id}: {count} queries migrated")


if __name__ == "__main__":
    main()
