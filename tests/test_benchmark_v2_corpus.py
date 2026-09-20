# Objective: Tests for the benchmark v2 corpus build: determinism, paired-variant invariants and partition targets.
"""The corpus is only useful if the paired design actually holds.

The central invariant is that a derived item shares its gold answer with the
canonical item it came from. If that ever breaks, an accuracy gap between a base
and its variant stops being evidence about wording and becomes a bug.
"""

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

from scripts.benchmark_v2 import build_corpus  # noqa: E402
from scripts.benchmark_v2.anchors import FACTS, render_documents  # noqa: E402
from scripts.benchmark_v2.graders import grade, reference_answer  # noqa: E402
from scripts.benchmark_v2.schema import normalize_text, validate_corpus  # noqa: E402

SEED = 42


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Build the corpus once and reuse it across the module."""
    out = tmp_path_factory.mktemp("corpus")
    items = build_corpus.build(SEED, out)
    return {"items": items, "out": out, "by_id": {item.id: item for item in items}}


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_corpus_size_is_within_the_requested_range(corpus):
    assert 300 <= len(corpus["items"]) <= 400


def test_every_discipline_is_substantially_represented(corpus):
    counts = {}
    for item in corpus["items"]:
        counts[item.discipline] = counts.get(item.discipline, 0) + 1
    assert set(counts) == {"matematica", "organizacao_computadores", "projeto_bd"}
    assert min(counts.values()) >= 0.25 * len(corpus["items"])


def test_semantic_outliers_hold_the_requested_share(corpus):
    outliers = [i for i in corpus["items"] if i.partition == "semantic_outlier"]
    share = len(outliers) / len(corpus["items"])
    assert 0.15 <= share <= 0.20, f"share fora da faixa pedida: {share:.3f}"


def test_the_corpus_is_overwhelmingly_machine_verifiable(corpus):
    """The legacy catalog had 1 verifiable item in 4780; this one must be the opposite."""
    verifiable = [i for i in corpus["items"] if i.answer_type != "rubric"]
    assert len(verifiable) / len(corpus["items"]) >= 0.95


def test_corpus_passes_its_own_validation_and_self_test(corpus):
    assert validate_corpus(corpus["items"]) == []
    assert build_corpus.self_test(corpus["items"]) == []


def test_no_two_items_share_a_statement(corpus):
    seen = {}
    for item in corpus["items"]:
        key = normalize_text(item.query)
        assert key not in seen, f"{item.id} repete o enunciado de {seen.get(key)}"
        seen[key] = item.id


# ---------------------------------------------------------------------------
# Paired design
# ---------------------------------------------------------------------------


def derived(corpus, partition):
    return [i for i in corpus["items"] if i.partition == partition]


@pytest.mark.parametrize("partition", ["verbosity_trap", "dense", "semantic_outlier"])
def test_derived_items_keep_the_gold_answer_of_their_base(corpus, partition):
    for item in derived(corpus, partition):
        base = corpus["by_id"][item.base_id]
        assert item.gold == base.gold, f"{item.id} divergiu do gabarito de {base.id}"
        assert item.grader == base.grader
        assert item.tolerance == base.tolerance


@pytest.mark.parametrize("partition", ["verbosity_trap", "dense", "semantic_outlier"])
def test_derived_items_actually_changed_the_statement(corpus, partition):
    for item in derived(corpus, partition):
        base = corpus["by_id"][item.base_id]
        assert normalize_text(item.query) != normalize_text(base.query)


def test_verbosity_traps_are_long_texts_over_one_step_logic(corpus):
    traps = derived(corpus, "verbosity_trap")
    assert traps
    for item in traps:
        base = corpus["by_id"][item.base_id]
        assert item.steps_required == base.steps_required == 1
        assert 150 <= item.distractor_tokens <= 420
        assert len(item.query.split()) >= 4 * len(base.query.split())


def test_dense_items_are_short_but_demand_more_reasoning(corpus):
    dense = derived(corpus, "dense")
    assert dense
    for item in dense:
        base = corpus["by_id"][item.base_id]
        assert item.steps_required >= 3
        assert len(item.query.split()) < len(base.query.split())


def test_outliers_cover_several_perturbation_strategies(corpus):
    strategies = {
        tag for item in derived(corpus, "semantic_outlier") for tag in item.tags if tag.startswith("perturbacao-")
    }
    assert len(strategies) >= 4


def test_perturbation_never_corrupts_a_number(corpus):
    """A typo inside a given value would change the answer and void the pairing.

    Order is not part of the invariant: the ``reorder`` perturbation moves the
    question ahead of the givens on purpose, so the multiset is what must match.
    """
    import re

    for item in derived(corpus, "semantic_outlier"):
        base = corpus["by_id"][item.base_id]
        assert sorted(re.findall(r"\d+", item.query)) == sorted(re.findall(r"\d+", base.query))


# ---------------------------------------------------------------------------
# RAG partitions
# ---------------------------------------------------------------------------


def test_anchored_items_point_at_a_document(corpus):
    anchored = derived(corpus, "rag_anchored")
    assert anchored
    for item in anchored:
        assert item.requires_rag
        assert item.anchor_doc_ids


def test_anchored_answers_are_present_in_their_document(corpus):
    """A gold value missing from the document would make retrieval impossible."""
    documents = render_documents()
    for fact in FACTS:
        text = documents[fact.doc]
        needle = fact.gold[0] if isinstance(fact.gold, list) else fact.gold
        if isinstance(needle, float):
            needle = f"{needle:g}".replace(".", ",")
        assert str(needle) in text, f"{fact.key}: gabarito ausente do documento {fact.doc}"


def test_abstention_items_have_no_gold_and_expect_a_refusal(corpus):
    abstention = [i for i in corpus["items"] if i.grader == "abstention"]
    assert abstention
    for item in abstention:
        assert item.expected_abstention
        assert item.gold is None
        assert item.requires_rag


def test_parametric_twins_need_no_document(corpus):
    twins = derived(corpus, "parametric_twin")
    assert twins
    for item in twins:
        assert not item.requires_rag
        assert not item.anchor_doc_ids


def test_every_anchor_document_is_written_to_disk(corpus):
    for doc_id in render_documents():
        path = corpus["out"] / "anchors" / f"{doc_id}.md"
        assert path.exists() and path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Determinism and outputs
# ---------------------------------------------------------------------------


def test_the_same_seed_reproduces_the_corpus_byte_for_byte(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    build_corpus.build(SEED, first)
    build_corpus.build(SEED, second)
    assert (first / "corpus.jsonl").read_bytes() == (second / "corpus.jsonl").read_bytes()


def test_a_different_seed_produces_a_different_corpus(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    build_corpus.build(SEED, first)
    build_corpus.build(SEED + 1, second)
    assert (first / "corpus.jsonl").read_bytes() != (second / "corpus.jsonl").read_bytes()


def test_jsonl_and_csv_agree_on_the_item_count(corpus):
    lines = [line for line in (corpus["out"] / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == len(corpus["items"])
    csv_lines = (corpus["out"] / "corpus.csv").read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) >= len(corpus["items"])  # header plus one row per item


def test_items_stay_readable_by_the_legacy_catalog_fields(corpus):
    """The old tooling reads id/query/theme/difficulty/lang/tags and must keep working."""
    for record in (json.loads(line) for line in (corpus["out"] / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if line):
        assert {"id", "query", "theme", "difficulty", "lang", "tags"} <= set(record)
        assert record["difficulty"] in {"easy", "medium", "hard"}
        assert record["theme"] == record["discipline"]


def test_every_item_is_graded_correctly_against_its_reference_answer(corpus):
    for item in corpus["items"]:
        if item.answer_type == "rubric":
            continue
        record = item.to_dict()
        assert grade(reference_answer(record), record).correct is True, item.id
