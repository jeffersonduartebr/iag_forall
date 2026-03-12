# Objective: Test coverage for router strategy behavior and regressions.
"""Test coverage for router strategy behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


import pytest
from unittest.mock import patch
from app.router_strategy import choose_top2_models

# Dados simulados do Bandit (Snapshot)
MOCK_SNAPSHOT = {
    "ollama/gemma3:4b": {"avg_latency": 0.5, "avg_cost": 0.0001},
    "openai/gpt-5.1":   {"avg_latency": 2.0, "avg_cost": 0.02},
    "ollama/llava:7b":  {"avg_latency": 3.0, "avg_cost": 0.0002},
}

# Qualidade simulada (Thompson Sampling)
MOCK_SAMPLED_QS = {
    "ollama/gemma3:4b": 6.0,
    "openai/gpt-5.1": 9.5,
    "ollama/llava:7b": 7.0
}

@pytest.fixture
def mock_bandit_data():
    """Execute the mock bandit data routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    with patch("app.router_strategy.get_snapshot", return_value=MOCK_SNAPSHOT), \
         patch("app.router_strategy.sample_metrics_from_snapshot", return_value=MOCK_SAMPLED_QS):
        yield

def test_router_prefers_quality(mock_bandit_data):
    """Se o peso da qualidade for alto, deve escolher o GPT-5."""
    candidates = ["ollama/gemma3:4b", "openai/gpt-5.1"]
    weights = {"w_quality": 100.0, "w_latency": 0.0, "w_cost": 0.0}
    
    top2 = choose_top2_models(candidates, weights, "query", modality="text")
    assert top2[0] == "openai/gpt-5.1"

def test_router_prefers_cost(mock_bandit_data):
    """Se o peso do custo for alto, deve escolher o Gemma."""
    candidates = ["ollama/gemma3:4b", "openai/gpt-5.1"]
    weights = {"w_quality": 0.0, "w_latency": 0.0, "w_cost": 100.0}
    
    top2 = choose_top2_models(candidates, weights, "query", modality="text")
    assert top2[0] == "ollama/gemma3:4b"

def test_vision_filter_logic(mock_bandit_data):
    """Se a modalidade for visão, deve filtrar modelos de texto."""
    candidates = ["ollama/gemma3:4b", "ollama/llava:7b"]
    weights = {"w_quality": 1.0, "w_latency": 1.0, "w_cost": 1.0}
    
    top2 = choose_top2_models(candidates, weights, "img", modality="vision")
    
    assert "ollama/llava:7b" in top2
    assert "ollama/gemma3:4b" not in top2

def test_uncertainty_trigger(mock_bandit_data):
    """Se a incerteza for alta, deve penalizar modelos locais (risk_factor)."""
    candidates = ["ollama/gemma3:4b", "openai/gpt-5.1"]
    weights = {"w_quality": 10.0, "w_latency": 0.0, "w_cost": 0.0}
    
    # Incerteza alta (0.9) > Threshold padrão (0.45)
    # O código reduz o score do local (0.6x) e aumenta o SOTA (1.3x)
    top2 = choose_top2_models(candidates, weights, "query", modality="text", uncertainty_score=0.9)
    
    assert top2[0] == "openai/gpt-5.1"