# Objective: Test coverage for golden sets behavior and regressions.
"""Tests for built-in golden sets and gate evaluation helpers."""

from app import golden_sets as gs


def test_list_and_get_golden_sets_roundtrip():
    """Golden-set listing and lookup should expose stable metadata and copied items."""
    listed = gs.list_golden_sets()
    assert listed
    assert listed[0]["id"] == "education_core_v1"
    assert listed[0]["item_count"] == 4

    golden = gs.get_golden_set("education_core_v1")
    assert golden is not None
    assert len(golden["items"]) == 4

    golden["items"][0]["prompt"] = "mutated"
    fresh = gs.get_golden_set("education_core_v1")
    assert fresh["items"][0]["prompt"] != "mutated"


def test_get_golden_set_and_prompts_handle_missing_ids():
    """Missing golden-set identifiers should degrade cleanly."""
    assert gs.get_golden_set("missing") is None
    assert gs.get_golden_set_prompts("missing") == []


def test_get_golden_set_prompts_filters_blank_prompts(monkeypatch):
    """Prompt extraction should skip blank prompt items."""
    monkeypatch.setitem(
        gs.GOLDEN_SETS,
        "tmp",
        {
            "id": "tmp",
            "title": "Tmp",
            "description": "Tmp",
            "gates": {},
            "items": [{"prompt": "One"}, {"prompt": "   "}, {"prompt": "Two"}],
        },
    )
    try:
        assert gs.get_golden_set_prompts("tmp") == ["One", "Two"]
    finally:
        gs.GOLDEN_SETS.pop("tmp", None)


def test_evaluate_golden_set_gate_pass_and_fail_paths():
    """Gate evaluation should compute rates and final pass/fail decisions."""
    gates = {
        "quality_mean_min": 5.5,
        "unsupported_rate_max": 0.2,
        "abstain_rate_max": 0.35,
        "grounded_rate_min": 0.25,
    }
    passing = gs.evaluate_golden_set_gate(
        [
            {"quality": 8.0, "metadata": {"verification_status": "supported", "abstained": False, "grounded": True}},
            {"quality": 6.0, "metadata": {"verification_status": "weakly_supported", "abstained": False, "grounded": False}},
            {"quality": 7.0, "metadata": {"verification_status": "supported", "abstained": False, "grounded": True}},
            {"quality": 6.0, "metadata": {"verification_status": "supported", "abstained": False, "grounded": False}},
        ],
        gates=gates,
    )
    assert passing["passed"] is True
    assert passing["metrics"]["n"] == 4

    failing = gs.evaluate_golden_set_gate(
        [
            {"quality": 2.0, "metadata": {"verification_status": "unsupported", "abstained": True, "grounded": False}},
            {"quality": 3.0, "metadata": {"verification_status": "unsupported", "abstained": True, "grounded": False}},
        ],
        gates=gates,
    )
    assert failing["passed"] is False
    assert failing["checks"]["quality_mean"] is False
    assert failing["checks"]["unsupported_rate"] is False
    assert failing["checks"]["abstain_rate"] is False
    assert failing["checks"]["grounded_rate"] is False
