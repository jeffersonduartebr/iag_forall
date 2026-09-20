# Objective: Tests for the benchmark v2 pre-classification pipeline: parsing, repair, resume and agreement stats.
"""The pre-classification is only trustworthy if its failure modes are.

A frontier model replying with prose around the JSON, or with an invented
label, must never silently become a data point. These tests pin the parser, the
single repair attempt, the resume logic and the cross-checks that compare the
classifier against what the generator already knows.
"""

import asyncio
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

from scripts.benchmark_v2 import agreement, preclassify  # noqa: E402

VALID = {
    "analise_inferencia": "dois passos",
    "analise_distratores": "nenhum",
    "dependencia_ancoragem": "NAO",
    "rotulo_complexidade_final": "MEDIA",
}

ITEM = {
    "id": "mat_c001",
    "query": "Quanto e 15% de 180?",
    "discipline": "matematica",
    "partition": "canonical",
    "complexity_gold": "BAIXA",
    "steps_required": 1,
    "requires_rag": False,
}


def call_returning(*payloads):
    """Build a call function that yields the given raw replies in order."""
    remaining = list(payloads)

    async def _call(model, prompt, hint=None):
        value = remaining.pop(0)
        if isinstance(value, Exception):
            raise value
        return value, {"cost_per_1k": 0.001, "prompt_tokens": 10, "completion_tokens": 20}

    return _call


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_extract_json_reads_a_bare_object():
    assert preclassify.extract_json(json.dumps(VALID))["rotulo_complexidade_final"] == "MEDIA"


def test_extract_json_reads_a_fenced_object():
    raw = f"Aqui esta a analise:\n```json\n{json.dumps(VALID)}\n```\nEspero ter ajudado."
    assert preclassify.extract_json(raw)["analise_inferencia"] == "dois passos"


def test_extract_json_rejects_a_reply_without_json():
    with pytest.raises(preclassify.SchemaError):
        preclassify.extract_json("O item parece de complexidade media.")


def test_extract_json_rejects_malformed_json():
    with pytest.raises(preclassify.SchemaError):
        preclassify.extract_json('{"analise_inferencia": "x",}')


def test_extract_json_rejects_a_top_level_list():
    with pytest.raises(preclassify.SchemaError):
        preclassify.extract_json("[1, 2, 3]")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_validation_requires_every_field():
    incomplete = {k: v for k, v in VALID.items() if k != "analise_distratores"}
    with pytest.raises(preclassify.SchemaError, match="analise_distratores"):
        preclassify.validate_classification(incomplete)


def test_validation_rejects_an_invented_label():
    with pytest.raises(preclassify.SchemaError, match="rotulo"):
        preclassify.validate_classification({**VALID, "rotulo_complexidade_final": "MUITO ALTA"})


def test_validation_rejects_an_invented_anchoring_verdict():
    with pytest.raises(preclassify.SchemaError, match="ancoragem"):
        preclassify.validate_classification({**VALID, "dependencia_ancoragem": "talvez"})


@pytest.mark.parametrize("value,expected", [("Sim", "SIM"), ("NÃO", "NAO"), ("nao", "NAO"), ("SIM.", "SIM")])
def test_validation_normalizes_the_anchoring_verdict(value, expected):
    result = preclassify.validate_classification({**VALID, "dependencia_ancoragem": value})
    assert result["dependencia_ancoragem"] == expected


def test_validation_normalizes_label_case():
    result = preclassify.validate_classification({**VALID, "rotulo_complexidade_final": " alta "})
    assert result["rotulo_complexidade_final"] == "ALTA"


# ---------------------------------------------------------------------------
# Classification with repair
# ---------------------------------------------------------------------------


def test_classification_succeeds_on_the_first_attempt():
    record = asyncio.run(preclassify.classify_one(ITEM, "m1", call_returning(json.dumps(VALID))))
    assert record["error"] is None
    assert record["attempts"] == 1
    assert record["rotulo_complexidade_final"] == "MEDIA"
    assert record["cost_usd"] == pytest.approx(0.001)


def test_a_malformed_reply_is_repaired_on_the_second_attempt():
    call = call_returning("desculpa, sem json", json.dumps(VALID))
    record = asyncio.run(preclassify.classify_one(ITEM, "m1", call))
    assert record["error"] is None
    assert record["attempts"] == 2


def test_two_malformed_replies_send_the_item_to_the_human_queue():
    call = call_returning("nada", "tambem nada")
    record = asyncio.run(preclassify.classify_one(ITEM, "m1", call))
    assert record["needs_human"] is True
    assert record["rotulo_complexidade_final"] is None
    assert "schema" in record["error"]


def test_a_provider_failure_is_recorded_and_not_retried_here():
    """The provider layer already retried; a failure reaching us is final."""
    call = call_returning(RuntimeError("502 bad gateway"), json.dumps(VALID))
    record = asyncio.run(preclassify.classify_one(ITEM, "m1", call))
    assert record["error"].startswith("RuntimeError")
    assert record["attempts"] == 1


def test_the_repair_attempt_receives_the_rejection_reason():
    seen = {}

    async def _call(model, prompt, hint=None):
        seen[len(seen)] = hint
        return (json.dumps(VALID) if hint else "sem json"), {"cost_per_1k": 0.0}

    asyncio.run(preclassify.classify_one(ITEM, "m1", _call))
    assert seen[0] is None
    assert seen[1] and "JSON" in seen[1]


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_completed_pairs_are_skipped_on_a_second_run(tmp_path):
    out = tmp_path / "pre.jsonl"
    asyncio.run(preclassify.run([ITEM], ["m1"], out, call_returning(json.dumps(VALID)), 2))
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 1

    # A call function with no payloads left would raise if it were used again.
    asyncio.run(preclassify.run([ITEM], ["m1"], out, call_returning(), 2))
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_failed_pairs_are_retried_on_a_second_run(tmp_path):
    out = tmp_path / "pre.jsonl"
    asyncio.run(preclassify.run([ITEM], ["m1"], out, call_returning("lixo", "lixo"), 2))
    assert preclassify.load_done(out) == set()

    asyncio.run(preclassify.run([ITEM], ["m1"], out, call_returning(json.dumps(VALID)), 2))
    assert ("mat_c001", "m1") in preclassify.load_done(out)


def test_every_item_model_pair_is_classified(tmp_path):
    items = [{**ITEM, "id": f"i{n}"} for n in range(3)]
    out = tmp_path / "pre.jsonl"
    payloads = [json.dumps(VALID)] * 6
    asyncio.run(preclassify.run(items, ["m1", "m2"], out, call_returning(*payloads), 3))
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert {(r["item_id"], r["model"]) for r in records} == {(f"i{n}", m) for n in range(3) for m in ("m1", "m2")}


# ---------------------------------------------------------------------------
# Agreement and cross-checks
# ---------------------------------------------------------------------------


def record(item_id, model, label, anchoring="NAO"):
    return {
        "item_id": item_id,
        "model": model,
        "rotulo_complexidade_final": label,
        "dependencia_ancoragem": anchoring,
        "cost_usd": 0.0,
    }


def test_classifier_agreement_lists_the_contested_items():
    records = [
        record("a", "m1", "BAIXA"),
        record("a", "m2", "BAIXA"),
        record("b", "m1", "ALTA"),
        record("b", "m2", "MEDIA"),
    ]
    result = agreement.classifier_agreement(records, "m1", "m2")
    assert result["n"] == 2
    assert [d["item_id"] for d in result["disagreements"]] == ["b"]


def test_agreement_with_the_generator_counts_overestimation():
    items = {"a": {"complexity_gold": "BAIXA"}, "b": {"complexity_gold": "BAIXA"}}
    records = [record("a", "m1", "ALTA"), record("b", "m1", "BAIXA")]
    result = agreement.agreement_with_generator(records, items, "m1")
    assert result["exact_match"] == 0.5
    assert result["overestimated_share"] == 0.5


def test_trap_effect_detects_padding_inflating_the_label():
    """The trap and its base differ only in padding, so a label change is the effect."""
    items = {
        "base": {"partition": "canonical", "complexity_gold": "BAIXA"},
        "trap": {"partition": "verbosity_trap", "base_id": "base", "complexity_gold": "BAIXA"},
    }
    records = [record("base", "m1", "BAIXA"), record("trap", "m1", "ALTA")]
    result = agreement.trap_effect(records, items)
    assert result["n_pairs"] == 1
    assert result["inflation_rate"] == 1.0


def test_trap_effect_is_zero_when_the_classifier_sees_through_the_padding():
    items = {
        "base": {"partition": "canonical"},
        "trap": {"partition": "verbosity_trap", "base_id": "base"},
    }
    records = [record("base", "m1", "BAIXA"), record("trap", "m1", "BAIXA")]
    assert agreement.trap_effect(records, items)["inflation_rate"] == 0.0


def test_anchoring_confusion_separates_the_two_error_directions():
    items = {"anchored": {"requires_rag": True}, "twin": {"requires_rag": False}}
    records = [record("anchored", "m1", "BAIXA", "NAO"), record("twin", "m1", "BAIXA", "SIM")]
    result = agreement.anchoring_confusion(records, items)
    assert result["false_negative"] == 1
    assert result["false_positive"] == 1
    assert result["accuracy"] == 0.0


def test_review_priority_puts_contested_items_first():
    items = {
        "agree": {"complexity_gold": "BAIXA", "partition": "canonical"},
        "contested": {"complexity_gold": "BAIXA", "partition": "canonical"},
        "vs_generator": {"complexity_gold": "ALTA", "partition": "canonical"},
    }
    records = [
        record("agree", "m1", "BAIXA"),
        record("agree", "m2", "BAIXA"),
        record("contested", "m1", "BAIXA"),
        record("contested", "m2", "ALTA"),
        record("vs_generator", "m1", "BAIXA"),
        record("vs_generator", "m2", "BAIXA"),
    ]
    ranked = agreement.review_priority(records, items, ["m1", "m2"])
    assert [item_id for item_id, _, _ in ranked][:2] == ["contested", "vs_generator"]
