# Objective: Tests for runtime query complexity detection.
"""Tests for query complexity detection and routing adjustments."""

from __future__ import annotations

from app.schemas import WorkloadHints
from app.services.query_complexity import (
    apply_complexity_runtime_adjustments,
    detect_query_complexity,
)


def test_detect_simple_short_question():
    assert detect_query_complexity("What is Python?", "simple_text") == "simple"


def test_detect_expert_system_design_query():
    query = (
        "Projete uma API de pagamentos idempotente para 50k TPS com sagas compensatórias, "
        "fencing tokens e particionamento multi-region active-active."
    )
    assert detect_query_complexity(query, "reasoning") == "expert"


def test_detect_high_compare_query():
    query = "Compare CQRS + event sourcing vs outbox pattern para integração com ERP legado em detalhes."
    assert detect_query_complexity(query, "reasoning") in {"high", "expert"}


def test_complexity_adjustments_prefer_cloud_for_expert():
    runtime_hints = {"interactive_priority": "high", "max_fallbacks": 2}
    adjusted = apply_complexity_runtime_adjustments(
        detected_complexity="expert",
        workload_class="reasoning",
        max_tokens=512,
        sync_deadline_seconds=70,
        provider_timeout_seconds=60,
        runtime_hints=runtime_hints,
        workload_hints=WorkloadHints(theme="programacao_desafios"),
    )
    assert adjusted["max_tokens"] >= 4096
    assert adjusted["runtime_hints"]["prefer_cloud_models"] is True
    assert adjusted["runtime_hints"]["benchmark_theme"] == "programacao_desafios"
    assert adjusted["runtime_hints"]["sync_deadline_seconds"] > 70
