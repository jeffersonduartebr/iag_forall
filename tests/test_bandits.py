# Objective: Test coverage for bandits behavior and regressions.
"""Test coverage for bandits behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


import pytest
from app.bandits import compute_reward, _dynamic_epsilon

def test_compute_reward_logic():
    """Testa se a função de recompensa respeita os limites [0, 1]."""
    # Caso ideal: Qualidade 10, Latência 0, Custo 0
    r_perfect = compute_reward("model", quality=10.0, latency_s=0.1, cost_per_1k=0.0)
    assert r_perfect > 0.8
    assert r_perfect <= 1.0

    # Caso ruim: Qualidade 0, Latência alta
    r_bad = compute_reward("model", quality=0.0, latency_s=50.0, cost_per_1k=1.0)
    assert r_bad < 0.5
    assert r_bad >= 0.0

def test_dynamic_epsilon():
    """Testa se a taxa de exploração aumenta quando há poucos dados."""
    # Poucos dados -> Epsilon maior
    stats_empty = {}
    eps_high = _dynamic_epsilon(stats_empty)
    
    # Muitos dados -> Epsilon padrão (menor)
    stats_full = {
        "m1": {"count": 100, "var": 0.01},
        "m2": {"count": 100, "var": 0.01}
    }
    eps_low = _dynamic_epsilon(stats_full)
    
    assert eps_high > eps_low