# Objective: Unit tests for result reliability annotations.
"""services.query_reliability: confidence, band, verification, abstention and publication."""

import pytest

from app.services import query_reliability as qrel


def _score(**overrides):
    kwargs = dict(answered=True, grounded=False, retrieval_used=False, fallback_used=False, flagged=False)
    kwargs.update(overrides)
    return qrel.confidence_score(0.3, **kwargs)


def test_confidence_adjustments():
    assert _score() == pytest.approx(0.7)
    assert _score(grounded=True) == pytest.approx(0.9)
    assert _score(retrieval_used=True) == pytest.approx(0.5)
    assert _score(fallback_used=True, flagged=True) == pytest.approx(0.5)
    assert _score(answered=False) == pytest.approx(0.1)
    assert (
        qrel.confidence_score(
            0.0, answered=True, grounded=True, retrieval_used=False, fallback_used=False, flagged=False
        )
        == 1.0
    )


def test_band_and_verification_thresholds():
    assert [qrel.confidence_band(s) for s in (0.75, 0.74, 0.45, 0.44)] == ["high", "medium", "medium", "low"]
    assert qrel.verification_status(answered=False, grounded=True, score=0.9) == "unsupported"
    assert qrel.verification_status(answered=True, grounded=True, score=0.7) == "supported"
    assert qrel.verification_status(answered=True, grounded=True, score=0.2) == "weakly_supported"
    assert qrel.verification_status(answered=True, grounded=False, score=0.44) == "unsupported"


def test_abstain_reasons():
    base = dict(
        answered=True,
        band="low",
        verification="unsupported",
        workload_class="chat",
        complexity="",
        retrieval_used=False,
        grounded=False,
        score=0.3,
    )
    assert qrel.abstain_reason(**{**base, "answered": False}) == "empty_answer"
    assert qrel.abstain_reason(**{**base, "workload_class": "reasoning"}) == "low_confidence"
    assert qrel.abstain_reason(**{**base, "complexity": "expert"}) == "low_confidence"
    assert qrel.abstain_reason(**base) is None
    evidence = {**base, "band": "medium", "workload_class": "knowledge_lookup", "retrieval_used": True, "score": 0.5}
    assert qrel.abstain_reason(**evidence) == "insufficient_evidence"


def test_enrich_replaces_unsupported_answer_and_flags_review():
    result = {"answer": "talvez", "metadata": {"uncertainty_score": 0.9, "workload_class": "reasoning"}}
    out = qrel.enrich_result_reliability(result)
    assert out["abstained"] is True and out["answer"] == qrel.SAFE_ABSTAIN_ANSWER
    assert out["review_status"] == "needs_review" and out["metadata"]["abstain_reason"] == "low_confidence"


def test_enrich_tool_turn_skips_abstention():
    result = {"answer": "", "tool_calls": [{"id": "1"}], "metadata": {"grounded": 1, "citations": ({"doc_id": "d"},)}}
    out = qrel.enrich_result_reliability(result)
    assert out["abstained"] is False and out["review_status"] == "auto_approved"
    assert out["grounded"] is True and out["citations"] == [{"doc_id": "d"}]
