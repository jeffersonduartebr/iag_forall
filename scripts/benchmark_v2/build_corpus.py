#!/usr/bin/env python3
# Objective: Build the benchmark v2 corpus (canonical items, paired variants, RAG anchors) as JSONL/CSV.
"""Build the ARISTO benchmark v2 corpus.

Usage::

    python3 -m scripts.benchmark_v2.build_corpus            # data/benchmark_v2/
    python3 -m scripts.benchmark_v2.build_corpus --seed 7 --out /tmp/corpus

The build is deterministic: the same seed produces byte-identical output. Every
generated item is self-tested — its own grader is run against its gold value and
the item is rejected if it fails, so a broken generator cannot reach the corpus.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

if __package__ in (None, ""):  # executed as a file rather than a module
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.benchmark_v2"

# Importing the generator modules populates the registry.
from . import generators_db, generators_hardware, generators_math  # noqa: E402,F401
from .anchors import ABSENT_QUESTIONS, FACTS, TWINS, render_documents  # noqa: E402
from .basegen import BaseSpec, generators_for  # noqa: E402
from .graders import grade, reference_answer  # noqa: E402
from .schema import (  # noqa: E402
    DISCIPLINES,
    Item,
    make_item,
    normalize_text,
    validate_corpus,
    write_jsonl,
)
from .transforms import pad_with_distractors, perturb_semantics  # noqa: E402

LOG = logging.getLogger("benchmark_v2.build")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "benchmark_v2"

PREFIX = {"matematica": "mat", "organizacao_computadores": "org", "projeto_bd": "bd"}

#: Target composition. The anchored and twin counts are fixed by the number of
#: institutional facts available; the rest is drawn to hit these numbers.
TARGETS = {
    "canonical": 120,
    "verbosity_trap": 60,
    "dense": 48,
    "semantic_outlier": 65,
}

MAX_DRAW_ATTEMPTS = 60


# ---------------------------------------------------------------------------
# Canonical pool
# ---------------------------------------------------------------------------


def build_canonical(rng: random.Random, per_discipline: int) -> Tuple[List[Item], Dict[str, BaseSpec]]:
    """Draw distinct canonical items, round-robin across each discipline's generators."""
    items: List[Item] = []
    specs: Dict[str, BaseSpec] = {}
    seen: set[str] = set()

    for discipline in DISCIPLINES:
        generators = generators_for(discipline)
        if not generators:
            raise RuntimeError(f"nenhum gerador registrado para {discipline}")
        produced = 0
        index = 0
        exhausted = 0
        while produced < per_discipline:
            if exhausted >= len(generators):
                raise RuntimeError(
                    f"{discipline}: geradores esgotados com {produced}/{per_discipline} itens distintos"
                )
            generator = generators[index % len(generators)]
            index += 1
            spec = None
            for _ in range(MAX_DRAW_ATTEMPTS):
                candidate = generator(rng)
                key = normalize_text(candidate.query)
                if key not in seen:
                    seen.add(key)
                    spec = candidate
                    break
            if spec is None:
                exhausted += 1
                LOG.warning("gerador %s esgotou as variacoes distintas", generator.__name__)
                continue
            exhausted = 0
            produced += 1
            item_id = f"{PREFIX[discipline]}_c{produced:03d}"
            items.append(
                make_item(
                    item_id=item_id,
                    query=spec.query,
                    discipline=discipline,
                    topic=spec.topic,
                    answer_type=spec.answer_type,
                    grader=spec.grader,
                    steps_required=spec.steps,
                    gold=spec.gold,
                    tolerance=spec.tolerance,
                    rubric=spec.rubric,
                    tags=list(spec.tags),
                    provenance=f"generated:{generator.__name__}",
                )
            )
            specs[item_id] = spec
    return items, specs


# ---------------------------------------------------------------------------
# Derived partitions
# ---------------------------------------------------------------------------


def build_verbosity_traps(bases: List[Item], rng: random.Random, target: int) -> List[Item]:
    """Bury one-step items in irrelevant narrative, keeping the gold answer."""
    pool = [item for item in bases if item.steps_required == 1 and item.answer_type != "rubric"]
    rng.shuffle(pool)
    out: List[Item] = []
    for n, base in enumerate(pool[:target], start=1):
        padded, added = pad_with_distractors(base.query, base.discipline, rng)
        out.append(
            make_item(
                item_id=f"{PREFIX[base.discipline]}_v{n:03d}",
                query=padded,
                discipline=base.discipline,
                topic=base.topic,
                answer_type=base.answer_type,
                grader=base.grader,
                steps_required=base.steps_required,
                gold=base.gold,
                tolerance=base.tolerance,
                partition="verbosity_trap",
                base_id=base.id,
                distractor_tokens=added,
                tags=list(base.tags),
                provenance=f"transform:pad_with_distractors:{base.id}",
            )
        )
    if len(out) < target:
        LOG.warning("apenas %d armadilhas de verbosidade possiveis (alvo %d)", len(out), target)
    return out


def build_dense(bases: List[Item], specs: Dict[str, BaseSpec], rng: random.Random, target: int) -> List[Item]:
    """Use the generator's terse phrasing: short text, more inference."""
    pool = [item for item in bases if specs[item.id].has_dense]
    rng.shuffle(pool)
    out: List[Item] = []
    for n, base in enumerate(pool[:target], start=1):
        spec = specs[base.id]
        out.append(
            make_item(
                item_id=f"{PREFIX[base.discipline]}_d{n:03d}",
                query=spec.query_dense or spec.query,
                discipline=base.discipline,
                topic=base.topic,
                answer_type=base.answer_type,
                grader=base.grader,
                steps_required=spec.steps_dense,
                gold=base.gold,
                tolerance=base.tolerance,
                partition="dense",
                base_id=base.id,
                tags=list(base.tags),
                provenance=f"generator_dense:{base.id}",
            )
        )
    if len(out) < target:
        LOG.warning("apenas %d itens densos possiveis (alvo %d)", len(out), target)
    return out


#: Share of outliers drawn from multi-step bases. Perturbing a one-step item
#: measures robustness alone; perturbing a four-step item measures robustness
#: *and* reasoning under noise, which discriminates far more between models.
OUTLIER_MULTISTEP_SHARE = 0.7


def build_outliers(bases: List[Item], rng: random.Random, target: int) -> List[Item]:
    """Perturb the surface form so the statement falls off the centroid distribution."""
    usable = [item for item in bases if item.answer_type != "rubric"]
    multi = [item for item in usable if item.steps_required > 1]
    single = [item for item in usable if item.steps_required == 1]
    rng.shuffle(multi)
    rng.shuffle(single)

    wanted_multi = min(len(multi), round(target * OUTLIER_MULTISTEP_SHARE))
    pool = multi[:wanted_multi] + single[: target - wanted_multi]
    rng.shuffle(pool)

    out: List[Item] = []
    for n, base in enumerate(pool[:target], start=1):
        text, strategy = perturb_semantics(base.query, rng)
        tags = list(base.tags) + [f"perturbacao-{strategy}"]
        out.append(
            make_item(
                item_id=f"{PREFIX[base.discipline]}_o{n:03d}",
                query=text,
                discipline=base.discipline,
                topic=base.topic,
                answer_type=base.answer_type,
                grader=base.grader,
                steps_required=base.steps_required,
                gold=base.gold,
                tolerance=base.tolerance,
                partition="semantic_outlier",
                base_id=base.id,
                tags=tags,
                lang="pt-en" if strategy == "code_switch" else "pt",
                provenance=f"transform:perturb_semantics:{strategy}:{base.id}",
            )
        )
    if len(out) < target:
        LOG.warning("apenas %d outliers possiveis (alvo %d)", len(out), target)
    return out


# ---------------------------------------------------------------------------
# RAG partitions
# ---------------------------------------------------------------------------


def build_anchored() -> List[Item]:
    """Items answerable only from the institutional documents, plus the abstention set."""
    out: List[Item] = []
    for n, fact in enumerate(FACTS, start=1):
        out.append(
            make_item(
                item_id=f"{PREFIX[fact.discipline]}_r{n:03d}",
                query=fact.question,
                discipline=fact.discipline,
                topic="ancoragem-institucional",
                answer_type=fact.answer_type,
                grader=fact.grader,
                steps_required=fact.steps,
                gold=fact.gold,
                tolerance=fact.tolerance,
                partition="rag_anchored",
                requires_rag=True,
                anchor_doc_ids=[fact.doc],
                tags=["rag", fact.doc],
                provenance=f"anchor:{fact.key}",
                contamination_risk="none",
            )
        )
    for n, absent in enumerate(ABSENT_QUESTIONS, start=1):
        out.append(
            make_item(
                item_id=f"{PREFIX[absent['discipline']]}_ra{n:03d}",
                query=absent["question"],
                discipline=absent["discipline"],
                topic="ancoragem-ausente",
                answer_type="exact",
                grader="abstention",
                steps_required=2,
                gold=None,
                partition="rag_anchored",
                requires_rag=True,
                expected_abstention=True,
                anchor_doc_ids=[absent["doc"]],
                tags=["rag", "abstencao", absent["doc"]],
                provenance=f"anchor_absent:{absent['key']}",
                contamination_risk="none",
            )
        )
    return out


def build_twins() -> List[Item]:
    """Parametric controls: the same question shape with no document behind it."""
    return [
        make_item(
            item_id=f"{PREFIX[twin.discipline]}_t{n:03d}",
            query=twin.question,
            discipline=twin.discipline,
            topic="conhecimento-parametrico",
            answer_type=twin.answer_type,
            grader=twin.grader,
            steps_required=twin.steps,
            gold=twin.gold,
            tolerance=twin.tolerance,
            partition="parametric_twin",
            tags=["parametrico"],
            provenance=f"twin:{twin.key}",
            contamination_risk="high",
        )
        for n, twin in enumerate(TWINS, start=1)
    ]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def self_test(items: List[Item]) -> List[str]:
    """Run every item's own grader against its gold answer.

    An item whose grader rejects its own reference answer is broken by
    definition, and would silently mark every model wrong.
    """
    failures: List[str] = []
    for item in items:
        record = item.to_dict()
        if item.answer_type == "rubric":
            continue
        result = grade(reference_answer(record), record)
        if result.correct is not True:
            failures.append(f"[{item.id}] gabarito rejeitado pelo proprio verificador: {result.detail}")
    return failures


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(items: List[Item], out_dir: Path, seed: int) -> None:
    """Write corpus.jsonl, corpus.csv, the anchor documents and the manifest."""
    import pandas as pd
    import yaml

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(items, out_dir / "corpus.jsonl")

    frame = pd.DataFrame([item.to_dict() for item in items])
    frame["tags"] = frame["tags"].apply(lambda values: "|".join(values))
    frame["anchor_doc_ids"] = frame["anchor_doc_ids"].apply(lambda values: "|".join(values))
    frame["rubric"] = frame["rubric"].apply(lambda values: " | ".join(values) if values else "")
    frame.to_csv(out_dir / "corpus.csv", index=False)

    anchors_dir = out_dir / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, text in render_documents().items():
        (anchors_dir / f"{doc_id}.md").write_text(text, encoding="utf-8")

    by_partition = Counter(item.partition for item in items)
    by_discipline = Counter(item.discipline for item in items)
    by_complexity = Counter(item.complexity_gold for item in items)
    manifest = {
        "version": "2.0",
        "seed": seed,
        "total": len(items),
        "by_partition": dict(sorted(by_partition.items())),
        "by_discipline": dict(sorted(by_discipline.items())),
        "by_complexity_gold": dict(sorted(by_complexity.items())),
        "by_answer_type": dict(sorted(Counter(i.answer_type for i in items).items())),
        "outlier_share": round(by_partition["semantic_outlier"] / len(items), 4),
        "verifiable_share": round(sum(1 for i in items if i.answer_type != "rubric") / len(items), 4),
        "anchor_documents": sorted(render_documents()),
    }
    (out_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def build(seed: int, out_dir: Path) -> List[Item]:
    """Build and validate the whole corpus."""
    rng = random.Random(seed)
    per_discipline = TARGETS["canonical"] // len(DISCIPLINES)

    canonical, specs = build_canonical(rng, per_discipline)
    LOG.info("canonicos: %d", len(canonical))

    items: List[Item] = list(canonical)
    items += build_verbosity_traps(canonical, rng, TARGETS["verbosity_trap"])
    items += build_dense(canonical, specs, rng, TARGETS["dense"])
    items += build_outliers(canonical, rng, TARGETS["semantic_outlier"])
    items += build_anchored()
    items += build_twins()

    problems = validate_corpus(items)
    problems += self_test(items)
    if problems:
        for problem in problems[:40]:
            LOG.error("%s", problem)
        raise SystemExit(f"corpus invalido: {len(problems)} problema(s)")

    write_outputs(items, out_dir, seed)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera o corpus de benchmark v2 do ARISTO.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    items = build(args.seed, args.out)
    partitions = Counter(item.partition for item in items)
    print(f"{len(items)} itens escritos em {args.out}")
    for partition, count in sorted(partitions.items()):
        print(f"  {partition:18s} {count:4d}")


if __name__ == "__main__":
    main()
