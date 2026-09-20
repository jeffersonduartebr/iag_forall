# Objective: Item schema, vocabularies and validation for the ARISTO benchmark v2 corpus.
"""Schema of one benchmark v2 item.

The first six fields (``id``, ``query``, ``theme``, ``difficulty``, ``lang``,
``tags``) keep the exact meaning they have in the legacy catalog, so
``app.benchmark_catalog``, ``app.benchmark_splits`` and the locust file read the
new corpus without a single change. Everything else is new and exists for one
reason: making an answer checkable by a program instead of by a rubric.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

SCHEMA_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

DISCIPLINES = ("matematica", "organizacao_computadores", "projeto_bd")

#: A partition is *how* the question is posed, never *what* it asks. Every
#: partition but ``rag_anchored``/``parametric_twin`` is a transformation of a
#: canonical item and keeps its gold answer.
PARTITIONS = (
    "canonical",
    "verbosity_trap",
    "dense",
    "semantic_outlier",
    "rag_anchored",
    "parametric_twin",
)

#: Derived partitions share ``gold``/``grader`` with their ``base_id``.
DERIVED_PARTITIONS = ("verbosity_trap", "dense", "semantic_outlier")

ANSWER_TYPES = ("numeric", "exact", "mcq", "set", "sql", "schema", "rubric")

COMPLEXITY_LABELS = ("BAIXA", "MEDIA", "ALTA")

#: Legacy catalog vocabulary, kept so the old tooling can read the corpus.
DIFFICULTIES = ("easy", "medium", "hard")


def complexity_from_steps(steps: int) -> str:
    """Map the generator's step count onto the label the classifier must recover.

    Mirrors the criteria given to the pre-classifier: one logical step is BAIXA,
    two or three chained concepts is MEDIA, beyond that is ALTA.
    """
    if steps <= 1:
        return "BAIXA"
    if steps <= 3:
        return "MEDIA"
    return "ALTA"


def difficulty_from_complexity(label: str) -> str:
    """Legacy ``difficulty`` value equivalent to a complexity label."""
    return {"BAIXA": "easy", "MEDIA": "medium", "ALTA": "hard"}[label]


# ---------------------------------------------------------------------------
# Item
# ---------------------------------------------------------------------------


@dataclass
class Item:
    """One benchmark item, canonical or derived."""

    id: str
    query: str
    theme: str  # == discipline, for catalog compatibility
    difficulty: str
    lang: str
    tags: List[str]

    discipline: str
    partition: str
    topic: str
    answer_type: str
    grader: str
    steps_required: int
    complexity_gold: str

    gold: Any = None
    tolerance: Optional[float] = None
    base_id: Optional[str] = None
    requires_rag: bool = False
    anchor_doc_ids: List[str] = field(default_factory=list)
    expected_abstention: bool = False
    distractor_tokens: int = 0
    rubric: Optional[List[str]] = None
    provenance: str = "generated"
    contamination_risk: str = "low"
    version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_item(
    *,
    item_id: str,
    query: str,
    discipline: str,
    topic: str,
    answer_type: str,
    grader: str,
    steps_required: int,
    gold: Any = None,
    partition: str = "canonical",
    **extra: Any,
) -> Item:
    """Build an item, deriving the label/difficulty fields from ``steps_required``."""
    complexity = complexity_from_steps(steps_required)
    tags = list(extra.pop("tags", []) or [])
    if topic not in tags:
        tags.append(topic)
    return Item(
        id=item_id,
        query=query,
        theme=discipline,
        difficulty=difficulty_from_complexity(complexity),
        lang=extra.pop("lang", "pt"),
        tags=tags,
        discipline=discipline,
        partition=partition,
        topic=topic,
        answer_type=answer_type,
        grader=grader,
        steps_required=steps_required,
        complexity_gold=complexity,
        gold=gold,
        **extra,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Accent-free, case-free, whitespace-collapsed form, for duplicate detection."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(stripped.lower().split())


def validate_item(item: Item) -> List[str]:
    """Return the list of schema violations of one item (empty means valid)."""
    problems: List[str] = []
    if item.discipline not in DISCIPLINES:
        problems.append(f"disciplina desconhecida: {item.discipline}")
    if item.partition not in PARTITIONS:
        problems.append(f"particao desconhecida: {item.partition}")
    if item.answer_type not in ANSWER_TYPES:
        problems.append(f"answer_type desconhecido: {item.answer_type}")
    if item.complexity_gold not in COMPLEXITY_LABELS:
        problems.append(f"rotulo de complexidade invalido: {item.complexity_gold}")
    if item.difficulty not in DIFFICULTIES:
        problems.append(f"difficulty invalida: {item.difficulty}")
    if item.steps_required < 1:
        problems.append("steps_required tem de ser >= 1")
    if not item.query.strip():
        problems.append("enunciado vazio")
    if item.partition in DERIVED_PARTITIONS and not item.base_id:
        problems.append(f"particao derivada {item.partition} sem base_id")
    if item.answer_type == "rubric":
        if not item.rubric:
            problems.append("item de rubrica sem criterios")
    elif item.gold is None and not item.expected_abstention:
        problems.append("item verificavel sem gabarito")
    if item.requires_rag and not item.anchor_doc_ids:
        problems.append("item ancorado sem anchor_doc_ids")
    if item.answer_type == "numeric" and item.tolerance is None:
        problems.append("item numerico sem tolerancia")
    return problems


def validate_corpus(items: Iterable[Item]) -> List[str]:
    """Return corpus-wide violations: duplicate ids, duplicate texts, dangling bases."""
    problems: List[str] = []
    items = list(items)
    seen_ids: Dict[str, int] = {}
    seen_texts: Dict[str, str] = {}
    known_ids = {it.id for it in items}

    for item in items:
        problems.extend(f"[{item.id}] {p}" for p in validate_item(item))
        seen_ids[item.id] = seen_ids.get(item.id, 0) + 1
        key = normalize_text(item.query)
        if key in seen_texts:
            problems.append(f"[{item.id}] enunciado identico ao de {seen_texts[key]}")
        else:
            seen_texts[key] = item.id
        if item.base_id and item.base_id not in known_ids:
            problems.append(f"[{item.id}] base_id inexistente: {item.base_id}")

    problems.extend(f"id duplicado: {i}" for i, n in seen_ids.items() if n > 1)
    return problems


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_jsonl(items: Iterable[Item], path: Path) -> int:
    """Write items as JSONL, one compact object per line. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield each record of a JSONL file, skipping blank lines."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_items(path: Path) -> List[Item]:
    """Read a corpus file back into ``Item`` objects."""
    return [Item(**record) for record in read_jsonl(path)]
