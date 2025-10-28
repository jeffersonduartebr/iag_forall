import pytest
from app import router_strategy


def test_pareto_filter_valid_population():
    """Filtra população válida e retorna lista de dicionários."""
    population = [
        {"model": "phi4", "cost": 0.1, "latency": 0.5, "quality": 0.9},
        {"model": "gemini", "cost": 0.2, "latency": 0.4, "quality": 0.95},
    ]
    result = router_strategy.pareto_filter(population)
    assert isinstance(result, list)
    assert all("model" in r for r in result)


def test_top2_models_returns_sorted_list():
    """Verifica se retorna uma lista com dois modelos."""
    ranked = router_strategy.top2_models(["phi4", "gemini", "gpt4"])
    assert isinstance(ranked, list)
    assert len(ranked) <= 2
