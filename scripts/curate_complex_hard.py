#!/usr/bin/env python3
"""Replace template hard/complex catalog rows with curated queries and references."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_curated_queries import _generate_for_bucket  # noqa: E402

CATALOG_DIR = ROOT / "data" / "benchmark_queries"
MANIFEST_PATH = CATALOG_DIR / "manifest.yaml"

SKIP_THEMES = frozenset({"programacao_desafios", "adversarial", "multimodal"})

TARGET_BY_DIFFICULTY = {
    "easy": 40,
    "medium": 55,
    "hard": 40,
    "complex": 15,
}

HARD_TEMPLATES_PT: List[Tuple[str, str]] = [
    (
        "Analise criticamente o papel de {subtopic} em {title}: quais evidências sustentam o consenso e quais o contestam?",
        "Resposta deve contrastar interpretações e citar critérios de avaliação de evidência.",
    ),
    (
        "Quais limitações metodológicas aparecem ao estudar {subtopic} dentro de {title}? Proponha mitigação.",
        "Deve listar vieses/limites e estratégias de validação ou triangulação.",
    ),
    (
        "Discuta um debate contemporâneo sobre {subtopic} em {title} e posicione-se com argumentos explícitos.",
        "Deve apresentar ao menos duas correntes e critérios de julgamento.",
    ),
    (
        "Em que contextos {subtopic} deixa de ser um modelo útil em {title}? Justifique com exemplos.",
        "Deve explicitar condições de falha do modelo e casos-limite.",
    ),
    (
        "Sintetize implicações práticas de {subtopic} para políticas ou decisões em {title}.",
        "Deve ligar teoria a ação com trade-offs claros.",
    ),
]

HARD_TEMPLATES_EN: List[Tuple[str, str]] = [
    (
        "Critically assess the role of {subtopic} in {title}: which evidence supports consensus and which challenges it?",
        "Answer should contrast interpretations and state evidence criteria.",
    ),
    (
        "What methodological limitations arise when studying {subtopic} within {title}? Propose mitigations.",
        "Should list biases/limits and validation strategies.",
    ),
    (
        "Discuss a contemporary debate on {subtopic} in {title} and take a reasoned position.",
        "Should present at least two schools of thought and judgment criteria.",
    ),
    (
        "Under which conditions does {subtopic} stop being a useful model in {title}? Justify with examples.",
        "Should state failure conditions and edge cases.",
    ),
    (
        "Synthesize practical implications of {subtopic} for policy or decisions in {title}.",
        "Should connect theory to action with explicit trade-offs.",
    ),
]

COMPLEX_TEMPLATES_PT: List[Tuple[str, str]] = [
    (
        "Construa uma análise integrada de {subtopic} em {title} articulando causas, atores, consequências e trade-offs em múltiplas dimensões.",
        "Resposta estruturada com causalidade, stakeholders e custos/benefícios.",
    ),
    (
        "Projete um framework de avaliação para {subtopic} em {title} com critérios, riscos, indicadores e cenários plausíveis.",
        "Deve incluir métricas, riscos e pelo menos dois cenários.",
    ),
    (
        "Elabore ensaio crítico sobre {subtopic} em {title} articulando ética, economia, política e tecnologia.",
        "Deve explicitar conflitos normativos e consequências distributivas.",
    ),
    (
        "Formule três hipóteses concorrentes sobre {subtopic} em {title} e descreva como testá-las empiricamente.",
        "Cada hipótese com desenho de teste e métrica observável.",
    ),
    (
        "Compare abordagens rivais de {subtopic} em {title} e proponha critérios de escolha para contextos de alta incerteza.",
        "Deve comparar pressupostos, custos e robustez preditiva.",
    ),
]

COMPLEX_TEMPLATES_EN: List[Tuple[str, str]] = [
    (
        "Build an integrated analysis of {subtopic} in {title} linking causes, actors, consequences, and multi-dimensional trade-offs.",
        "Structured answer with causality, stakeholders, and cost/benefit framing.",
    ),
    (
        "Design an evaluation framework for {subtopic} in {title} with criteria, risks, indicators, and plausible scenarios.",
        "Must include metrics, risks, and at least two scenarios.",
    ),
    (
        "Write a critical essay on {subtopic} in {title} connecting ethics, economics, politics, and technology.",
        "Should surface normative conflicts and distributive consequences.",
    ),
    (
        "Formulate three competing hypotheses on {subtopic} in {title} and describe how to test them empirically.",
        "Each hypothesis needs a test design and observable metric.",
    ),
    (
        "Compare rival approaches to {subtopic} in {title} and propose selection criteria under high uncertainty.",
        "Should compare assumptions, costs, and predictive robustness.",
    ),
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _subtopic_label(theme_id: str, subtopic: str) -> str:
    return subtopic.replace("-", " ").replace("_", " ")


def _pick_lang(theme: Dict[str, Any], difficulty: str, index: int) -> str:
    default = str(theme.get("default_lang", "pt"))
    if default != "mixed":
        return default
    if difficulty == "complex":
        return "en" if index % 3 == 0 else "pt"
    return "pt" if index % 2 == 0 else "en"


def _generate_curated(
    theme: Dict[str, Any],
    difficulty: str,
    count: int,
    existing_norm: set[str],
    start_index: int,
) -> List[Dict[str, Any]]:
    theme_id = str(theme["id"])
    title = str(theme.get("title", theme_id))
    subtopics = list(theme.get("subtopics") or [theme_id])
    templates_pt = HARD_TEMPLATES_PT if difficulty == "hard" else COMPLEX_TEMPLATES_PT
    templates_en = HARD_TEMPLATES_EN if difficulty == "hard" else COMPLEX_TEMPLATES_EN
    generated: List[Dict[str, Any]] = []
    attempt = 0
    idx = start_index
    while len(generated) < count and attempt < count * 40:
        subtopic = subtopics[attempt % len(subtopics)]
        label = _subtopic_label(theme_id, subtopic)
        lang = _pick_lang(theme, difficulty, attempt)
        template_pool = templates_en if lang == "en" else templates_pt
        template_idx = (attempt // len(subtopics)) % len(template_pool)
        template, reference = template_pool[template_idx]
        query = template.format(subtopic=label, title=title)
        norm = _normalize(query)
        if norm in existing_norm:
            attempt += 1
            continue
        idx += 1
        generated.append(
            {
                "id": f"{theme_id}_{idx:03d}",
                "query": query,
                "theme": theme_id,
                "difficulty": difficulty,
                "lang": lang,
                "tags": [subtopic],
                "reference": reference,
                "criticality": "high" if difficulty == "complex" else "medium",
            }
        )
        existing_norm.add(norm)
        attempt += 1
    if len(generated) < count:
        raise RuntimeError(f"Could not curate enough {difficulty} rows for {theme_id}: {len(generated)}/{count}")
    return generated


def curate_theme_file(theme: Dict[str, Any]) -> Tuple[int, int]:
    theme_id = str(theme["id"])
    path = CATALOG_DIR / str(theme["file"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    buckets: Dict[str, List[Dict[str, Any]]] = {k: [] for k in TARGET_BY_DIFFICULTY}
    for row in rows:
        diff = str(row.get("difficulty", "medium"))
        if diff in buckets:
            buckets[diff].append(row)

    kept: List[Dict[str, Any]] = []
    for diff in ("easy", "medium"):
        kept.extend(buckets[diff][: TARGET_BY_DIFFICULTY[diff]])

    existing_norm = {_normalize(str(r.get("query", ""))) for r in kept}
    for diff in ("easy", "medium"):
        deficit = TARGET_BY_DIFFICULTY[diff] - sum(1 for r in kept if r.get("difficulty") == diff)
        if deficit > 0:
            generated = _generate_for_bucket(theme, diff, deficit, existing_norm, len(kept))
            kept.extend(generated)
            existing_norm.update(_normalize(str(r.get("query", ""))) for r in generated)

    hard_curated = _generate_curated(theme, "hard", TARGET_BY_DIFFICULTY["hard"], existing_norm, len(kept))
    complex_curated = _generate_curated(
        theme,
        "complex",
        TARGET_BY_DIFFICULTY["complex"],
        existing_norm,
        len(kept) + len(hard_curated),
    )

    merged = kept + hard_curated + complex_curated
    expected = sum(TARGET_BY_DIFFICULTY.values())
    if len(merged) != expected:
        raise RuntimeError(f"{theme_id}: expected {expected} rows after curation, got {len(merged)}")

    for i, row in enumerate(merged, start=1):
        row["id"] = f"{theme_id}_{i:03d}"

    with path.open("w", encoding="utf-8") as fh:
        for row in merged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(hard_curated), len(complex_curated)


def main() -> int:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    total_hard = 0
    total_complex = 0
    for theme in manifest["themes"]:
        theme_id = str(theme["id"])
        if theme_id in SKIP_THEMES:
            print(f"SKIP {theme_id}")
            continue
        hard_n, complex_n = curate_theme_file(theme)
        total_hard += hard_n
        total_complex += complex_n
        print(f"OK {theme_id}: hard={hard_n} complex={complex_n}")
    print(f"TOTAL curated hard={total_hard} complex={total_complex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
