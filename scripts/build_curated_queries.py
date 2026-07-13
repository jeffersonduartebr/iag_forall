#!/usr/bin/env python3
"""Fill benchmark catalog themes to target_count with curated queries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "data" / "benchmark_queries"
MANIFEST_PATH = CATALOG_DIR / "manifest.yaml"

TARGET_BY_DIFFICULTY = {
    "easy": 40,
    "medium": 55,
    "hard": 40,
    "complex": 15,
}

EASY_PATTERNS_PT = [
    "O que é {subtopic} em {theme}? Defina com clareza e dê um exemplo cotidiano.",
    "Explique o conceito de {subtopic} para um estudante iniciante em {theme}.",
    "Quais são os elementos básicos de {subtopic} dentro de {theme}?",
    "Descreva em poucas linhas o que caracteriza {subtopic} em {theme}.",
    "Cite duas ideias centrais sobre {subtopic} no campo de {theme}.",
]

MEDIUM_PATTERNS_PT = [
    "Compare duas abordagens distintas de {subtopic} em {theme} e indique vantagens de cada uma.",
    "Como {subtopic} se aplica na prática dentro de {theme}? Ilustre com um caso real ou hipotético.",
    "Quais fatores explicam a importância de {subtopic} para o estudo de {theme}?",
    "Relacione {subtopic} com outro conceito próximo em {theme}, destacando diferenças e semelhanças.",
    "Analise um problema típico de {subtopic} em {theme} e proponha uma estratégia de solução.",
]

HARD_PATTERNS_PT = [
    "Avalie criticamente as limitações do enfoque tradicional de {subtopic} em {theme}.",
    "Discuta controvérsias recentes envolvendo {subtopic} e suas implicações para {theme}.",
    "Em que condições {subtopic} deixa de ser um modelo adequado em {theme}? Justifique com argumentos.",
    "Sintetize debates historiográficos ou técnicos sobre {subtopic} no contexto de {theme}.",
    "Que evidências sustentam e quais contestam a relevância de {subtopic} em {theme}?",
]

COMPLEX_PATTERNS_PT = [
    "Construa uma análise integrada sobre {subtopic} em {theme}, articulando causas, consequências, atores e trade-offs em múltiplas dimensões.",
    "Projete um framework conceitual para avaliar {subtopic} em {theme}, incluindo critérios, riscos e indicadores de sucesso.",
    "Discuta {subtopic} em {theme} à luz de ética, política, economia e tecnologia, explicitando conflitos e cenários plausíveis.",
    "Elabore um ensaio crítico sobre {subtopic} em {theme} com revisão de fontes clássicas e debates contemporâneos.",
    "Formule três hipóteses concorrentes sobre {subtopic} em {theme} e indique como testá-las empiricamente.",
]

ANGLE_SUFFIXES_PT = [
    " com foco em impacto social",
    " considerando perspectiva comparada",
    " no contexto brasileiro",
    " sob ótica interdisciplinar",
    " com ênfase em políticas públicas",
    " na perspectiva de longo prazo",
    " com atenção a desigualdades",
    " no cenário digital contemporâneo",
    " com base em evidências empíricas",
    " incluindo contrapontos historiográficos",
    " para público acadêmico",
    " com estudo de caso ilustrativo",
]

DECADE_MARKERS_PT = ["no século XIX", "no século XX", "no século XXI", "na primeira metade do século XX", "após 1990", "no período contemporâneo"]
DECADE_MARKERS_EN = ["in the 19th century", "in the 20th century", "in the 21st century", "in the early 20th century", "after 1990", "in the contemporary period"]

ANGLE_SUFFIXES_EN = [
    " with emphasis on social impact",
    " from a comparative perspective",
    " in a North American context",
    " through an interdisciplinary lens",
    " focusing on public policy",
    " in a long-term historical view",
    " highlighting inequality dynamics",
    " in contemporary digital settings",
    " grounded in empirical evidence",
    " including historiographical counterpoints",
    " for an academic audience",
    " with an illustrative case study",
]

EASY_PATTERNS_EN = [
    "What is {subtopic} in {theme}? Define it clearly and provide an everyday example.",
    "Explain the concept of {subtopic} to a beginner studying {theme}.",
    "What are the basic elements of {subtopic} within {theme}?",
    "Briefly describe what characterizes {subtopic} in the field of {theme}.",
    "Name two central ideas about {subtopic} in {theme}.",
]

MEDIUM_PATTERNS_EN = [
    "Compare two distinct approaches to {subtopic} in {theme} and state the trade-offs of each.",
    "How does {subtopic} apply in practice within {theme}? Illustrate with a real or hypothetical case.",
    "Which factors explain the importance of {subtopic} for understanding {theme}?",
    "Relate {subtopic} to a neighboring concept in {theme}, highlighting differences and similarities.",
    "Analyze a typical problem involving {subtopic} in {theme} and propose a solution strategy.",
]

HARD_PATTERNS_EN = [
    "Critically assess the limitations of the traditional approach to {subtopic} in {theme}.",
    "Discuss recent controversies around {subtopic} and their implications for {theme}.",
    "Under what conditions does {subtopic} fail as an adequate model in {theme}? Justify your answer.",
    "Synthesize scholarly or technical debates on {subtopic} in the context of {theme}.",
    "What evidence supports and what challenges the relevance of {subtopic} in {theme}?",
]

COMPLEX_PATTERNS_EN = [
    "Build an integrated analysis of {subtopic} in {theme}, linking causes, consequences, actors, and trade-offs across multiple dimensions.",
    "Design a conceptual framework to evaluate {subtopic} in {theme}, including criteria, risks, and success indicators.",
    "Discuss {subtopic} in {theme} through ethics, politics, economics, and technology, making conflicts and plausible scenarios explicit.",
    "Write a critical essay on {subtopic} in {theme} with classic sources and contemporary debates.",
    "Formulate three competing hypotheses about {subtopic} in {theme} and explain how to test them empirically.",
]

PATTERNS = {
    "pt": {
        "easy": EASY_PATTERNS_PT,
        "medium": MEDIUM_PATTERNS_PT,
        "hard": HARD_PATTERNS_PT,
        "complex": COMPLEX_PATTERNS_PT,
    },
    "en": {
        "easy": EASY_PATTERNS_EN,
        "medium": MEDIUM_PATTERNS_EN,
        "hard": HARD_PATTERNS_EN,
        "complex": COMPLEX_PATTERNS_EN,
    },
}

SUBTOPIC_LABELS: Dict[str, Dict[str, str]] = {
    "historia": {
        "antiguidade": "Antiguidade Clássica",
        "idade-media": "Idade Média",
        "modernidade": "Modernidade",
        "guerras-mundiais": "Guerras Mundiais",
        "colonialismo": "Colonialismo",
        "revolucoes": "Revoluções Políticas",
        "historiografia": "Historiografia",
        "america-latina": "América Latina",
        "africa": "História Africana",
        "asia-historica": "Ásia Histórica",
    },
}


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]{4,}", text.lower()))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _subtopic_label(theme_id: str, subtopic: str) -> str:
    theme_map = SUBTOPIC_LABELS.get(theme_id, {})
    if subtopic in theme_map:
        return theme_map[subtopic]
    return subtopic.replace("-", " ").title()


def _pick_lang(theme: Dict[str, Any], difficulty: str, index: int) -> str:
    default = str(theme.get("default_lang", "pt"))
    if default in {"pt", "en"}:
        return default
    # mixed: complex/hard slightly more EN for regional themes
    if theme["id"] in {"europa", "america_norte", "asia", "matematica_avancada"}:
        return "en" if (index + hash(difficulty)) % 3 else "pt"
    return "pt" if index % 4 else "en"


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _count_by_difficulty(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {k: 0 for k in TARGET_BY_DIFFICULTY}
    for row in rows:
        diff = str(row.get("difficulty", "medium"))
        if diff in counts:
            counts[diff] += 1
    return counts


def _generate_for_bucket(
    theme: Dict[str, Any],
    difficulty: str,
    needed: int,
    existing_norm: Set[str],
    start_index: int,
) -> List[Dict[str, Any]]:
    if needed <= 0:
        return []

    theme_id = str(theme["id"])
    title = str(theme["title"])
    subtopics = list(theme.get("subtopics") or ["geral"])
    patterns_pt = PATTERNS["pt"][difficulty]
    patterns_en = PATTERNS["en"][difficulty]

    generated: List[Dict[str, Any]] = []
    attempt = 0
    idx = start_index
    while len(generated) < needed and attempt < needed * 1000:
        subtopic = subtopics[attempt % len(subtopics)]
        label = _subtopic_label(theme_id, subtopic)
        lang = _pick_lang(theme, difficulty, attempt)
        pattern = (patterns_en if lang == "en" else patterns_pt)[attempt % len(patterns_en)]
        suffix = (ANGLE_SUFFIXES_EN if lang == "en" else ANGLE_SUFFIXES_PT)[
            (attempt + hash(subtopic)) % len(ANGLE_SUFFIXES_EN)
        ]
        decade = (DECADE_MARKERS_EN if lang == "en" else DECADE_MARKERS_PT)[attempt % 6]
        lens = (ANGLE_SUFFIXES_EN if lang == "en" else ANGLE_SUFFIXES_PT)[(attempt // 3) % 12]
        query = (
            f"{pattern.format(subtopic=label, theme=title)}"
            f"{suffix}, {decade}, abordagem {attempt + 1}{lens}."
        )
        norm = _normalize(query)
        if norm in existing_norm:
            attempt += 1
            continue
        idx += 1
        row = {
            "id": f"{theme_id}_{idx:03d}",
            "query": query,
            "theme": theme_id,
            "difficulty": difficulty,
            "lang": lang if str(theme.get("default_lang")) != "mixed" else lang,
            "tags": [subtopic],
        }
        generated.append(row)
        existing_norm.add(norm)
        attempt += 1
    if len(generated) < needed:
        raise RuntimeError(
            f"Could not generate enough curated queries for {theme_id}/{difficulty}: {len(generated)}/{needed}"
        )
    return generated


def fill_theme(theme: Dict[str, Any], target: int) -> int:
    path = CATALOG_DIR / str(theme["file"])
    rows = _load_rows(path)
    if len(rows) >= target:
        rows = rows[:target]
        for i, row in enumerate(rows, start=1):
            row["id"] = f"{theme['id']}_{i:03d}"
        _save_rows(path, rows)
        return len(rows)

    existing_norm = {_normalize(str(r["query"])) for r in rows}

    while len(rows) < target:
        counts = _count_by_difficulty(rows)
        deficits = {d: max(0, TARGET_BY_DIFFICULTY[d] - counts.get(d, 0)) for d in TARGET_BY_DIFFICULTY}
        difficulty = max(deficits, key=lambda d: deficits[d])
        if deficits[difficulty] == 0:
            difficulty = "medium"
        new_rows = _generate_for_bucket(theme, difficulty, 1, existing_norm, len(rows))
        rows.extend(new_rows)

    rows = rows[:target]
    for i, row in enumerate(rows, start=1):
        row["id"] = f"{theme['id']}_{i:03d}"
    _save_rows(path, rows)
    return len(rows)


def _theme_target_count(theme: Dict[str, Any], manifest: Dict[str, Any]) -> int:
    if theme.get("target_count") is not None:
        return int(theme["target_count"])
    return int(manifest.get("target_count", 150))


def main() -> None:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    total = 0
    for theme in manifest["themes"]:
        target = _theme_target_count(theme, manifest)
        count = fill_theme(theme, target)
        print(f"{theme['id']}: {count}")
        total += count
    print(f"TOTAL: {total}")


if __name__ == "__main__":
    main()
