# Objective: Tests for the routing confusion matrix that separates waste from quality collapse.
"""What the matrix must and must not count.

The failure mode that would make the whole metric worthless is including the
~95% of traffic whose "quality" is ``proxy_quality`` — the bandit's own
posterior rescaled. A matrix built on that measures the router against its own
opinion of itself, and would show whatever the router already believes.
"""

from app.services.formative_observability import (
    QUALITY_FLOOR,
    classify_model,
    record_routing_cell,
    routing_verdict,
)


# ---------------------------------------------------------------------------
# Model classification
# ---------------------------------------------------------------------------


def test_ollama_models_are_local():
    assert classify_model("ollama/phi4:latest") == "local"
    assert classify_model("ollama/qwen2.5:0.5b") == "local"


def test_frontier_models_are_sota():
    assert classify_model("anthropic/claude-opus-4-8") == "sota"
    assert classify_model("openai/gpt-5.5-2026-04-23") == "sota"


def test_other_hosted_models_are_cloud():
    assert classify_model("anthropic/claude-haiku-4-5-20251001") == "cloud"


def test_an_unknown_model_is_not_treated_as_local():
    """Guessing 'local' for an unknown model would under-count waste."""
    assert classify_model("algum/modelo-desconhecido") == "cloud"


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def test_a_small_model_failing_a_hard_request_is_collapse():
    assert routing_verdict("expert", "local", 3.0, "judge") == "collapse"
    assert routing_verdict("high", "local", 5.9, "judge") == "collapse"


def test_a_small_model_succeeding_on_a_hard_request_is_not_collapse():
    """Collapse is about the outcome, not about the choice."""
    assert routing_verdict("expert", "local", 8.0, "judge") == "ok"


def test_a_frontier_model_on_a_simple_request_is_waste():
    assert routing_verdict("simple", "sota", 9.0, "judge") == "waste"
    assert routing_verdict("moderate", "cloud", 7.5, "judge") == "waste"


def test_a_frontier_model_that_also_failed_is_not_waste():
    """If the expensive answer was bad, the problem is not that it was expensive."""
    assert routing_verdict("simple", "sota", 4.0, "judge") == "ok"


def test_a_small_model_on_a_simple_request_is_ok():
    assert routing_verdict("simple", "local", 8.0, "judge") == "ok"


def test_a_frontier_model_on_a_hard_request_is_ok():
    assert routing_verdict("expert", "sota", 9.0, "judge") == "ok"


def test_the_quality_floor_is_the_boundary():
    assert routing_verdict("high", "local", QUALITY_FLOOR - 0.01, "judge") == "collapse"
    assert routing_verdict("high", "local", QUALITY_FLOOR, "judge") == "ok"


# ---------------------------------------------------------------------------
# The guard that makes the metric meaningful
# ---------------------------------------------------------------------------


def test_unjudged_traffic_is_always_inconclusive():
    """proxy_quality is the bandit's own output; counting it measures nothing."""
    assert routing_verdict("simple", "sota", 9.0, "bandit_proxy") == "inconclusive"
    assert routing_verdict("expert", "local", 2.0, "bandit_proxy") == "inconclusive"


def test_the_length_heuristic_is_also_inconclusive():
    assert routing_verdict("simple", "sota", 9.0, "heuristic_fallback") == "inconclusive"


def test_a_missing_complexity_is_inconclusive():
    assert routing_verdict(None, "sota", 9.0, "judge") == "inconclusive"
    assert routing_verdict("", "local", 2.0, "judge") == "inconclusive"


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def test_recording_returns_the_cell():
    cell = record_routing_cell("ollama/phi4:latest", "expert", 3.0, "judge")
    assert cell["model_class"] == "local"
    assert cell["verdict"] == "collapse"
    assert cell["detected_complexity"] == "expert"


def test_a_missing_complexity_is_labelled_unknown_not_dropped():
    cell = record_routing_cell("ollama/phi4:latest", None, 3.0, "judge")
    assert cell["detected_complexity"] == "unknown"
    assert cell["verdict"] == "inconclusive"


def test_recording_never_raises_on_an_odd_model_name():
    """Metrics are best-effort; they must not take a request down with them."""
    assert record_routing_cell("", "simple", 7.0, "judge")["model_class"] == "cloud"


def test_every_verdict_is_reachable():
    cells = [
        record_routing_cell("ollama/phi4:latest", "expert", 2.0, "judge"),
        record_routing_cell("anthropic/claude-opus-4-8", "simple", 9.0, "judge"),
        record_routing_cell("ollama/phi4:latest", "simple", 9.0, "judge"),
        record_routing_cell("ollama/phi4:latest", "simple", 9.0, "bandit_proxy"),
    ]
    assert {c["verdict"] for c in cells} == {"collapse", "waste", "ok", "inconclusive"}


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def test_the_feedback_payload_carries_the_detected_complexity():
    """Without this the matrix has no complexity axis at all."""
    import inspect

    from app.services import query_runtime

    source = inspect.getsource(query_runtime.record_query_side_effects)
    assert '"detected_complexity"' in source


def test_persist_log_emits_the_cell():
    import inspect

    from app.services import feedback_stages

    source = inspect.getsource(feedback_stages.persist_log)
    assert "record_routing_cell" in source
    assert "quietly" in source
