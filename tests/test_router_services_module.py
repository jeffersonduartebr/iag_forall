# Objective: Test coverage for router services module behavior and regressions.
"""Tests for shared router service helpers."""

import pytest

from app.services import router_services as rs


def test_normalize_modality_and_final_prompt_paths():
    """Modality normalization and prompt construction should cover RAG and non-RAG branches."""
    assert rs.normalize_modality("text", "img") == "vision"
    assert rs.normalize_modality("", None) == "text"

    assert rs.build_final_prompt("Q", "SYS", use_rag=False, rag_text=None) == "SYS\n\nUsuário: Q"
    assert rs.build_final_prompt("Q", "", use_rag=False, rag_text=None) == "Q"
    assert rs.build_final_prompt("Q", "SYS", use_rag=True, rag_text=None) == "SYS\n\nUsuário: Q"
    assert rs.build_final_prompt("Q", "", use_rag=True, rag_text="CTX") == "CTX"


def test_parse_meta_cost_uses_explicit_and_fallback_cost_paths():
    """Metadata parsing should handle explicit provider cost, lookup fallback, and non-dict metadata."""
    meta = {"load_time": 1.2, "prompt_tokens": 10, "completion_tokens": 20, "cost_per_1k": 0.5}
    parsed = rs.parse_meta_cost(meta, "ollama/test", lambda *_: 999.0)
    assert parsed[:4] == (10, 20, 0.5, 1.2)

    parsed_lookup = rs.parse_meta_cost(
        {"load_time": 0.2, "prompt_tokens": 3, "completion_tokens": 4},
        "ollama/test",
        lambda model, p, c: 1.23,
    )
    assert parsed_lookup[:4] == (3, 4, 1.23, 0.2)

    parsed_error = rs.parse_meta_cost(
        {"prompt_tokens": 1, "completion_tokens": 2},
        "ollama/test",
        lambda *_: (_ for _ in ()).throw(RuntimeError("pricing down")),
    )
    assert parsed_error[:3] == (1, 2, 0.0)

    parsed_other = rs.parse_meta_cost("bad-meta", "ollama/test", lambda *_: 0.0)
    assert parsed_other == (0, 0, 0.0, 0.0, {})


def test_should_enable_dedup_and_compute_judge_probability():
    """Dedup and judging helpers should cover settings and model-specific discounts."""
    assert rs.should_enable_dedup(lambda *_: "1", True) is True
    assert rs.should_enable_dedup(lambda *_: "0", True) is False
    assert rs.should_enable_dedup(lambda *_: "1", False) is False

    assert rs.compute_judge_probability(2, 0.2, "ollama/test", 0.05) == 1.0
    assert rs.compute_judge_probability(100, 0.8, "gpt-4o", 0.05) == pytest.approx(0.08)
    assert rs.compute_judge_probability(100, 0.01, "ollama/test", 0.05) >= 0.05
