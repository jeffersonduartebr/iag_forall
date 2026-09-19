# Objective: Exact-value tests for the bandit policies (epsilon-greedy, UCB1, Thompson, meta vote).
"""services.bandit_policy against the textbook formulas.

The property tests only check invariants (the choice is a candidate, an
unvisited arm wins UCB1); these pin the numbers, so a changed constant or
comparison is caught (mutation testing, see scripts/run_mutation.sh).
"""

import math

import numpy as np
import pytest
from app.services import bandit_policy as bp
from hypothesis import given
from hypothesis import strategies as st


@pytest.mark.parametrize(
    ("stats", "eps", "expected"),
    [
        ({}, 0.10, 0.25),  # sem estatísticas: +0,15
        ({}, 0.95, 1.0),  # teto em 1
        ({"a": {"count": 5, "var": 0.0}}, 0.10, 0.10),
        ({"a": {"count": 2, "var": 0.0}}, 0.10, 0.20),  # braço com < 3 amostras: +0,10
        ({"a": {"count": 3, "var": 0.0}}, 0.10, 0.10),  # exatamente 3 não soma
        ({"a": {"count": 5, "var": 0.10}}, 0.10, 0.18),  # variância média > 0,05: +0,08
        ({"a": {"count": 5, "var": 0.05}}, 0.10, 0.10),  # exatamente 0,05 não soma
        ({"a": {"count": 1, "var": 0.10}}, 0.10, 0.28),
        ({"a": {}}, 0.10, 0.20),  # sem count: conta 0
        ({"a": {"count": 9, "var": 0.0}, "b": {"count": 9, "var": 0.2}}, 0.10, 0.18),  # média 0,1
        ({"a": {"count": 5}}, -0.50, 0.0),  # piso em 0
        ({"a": {"count": 0, "var": 1.0}}, 0.95, 1.0),
    ],
)
def test_dynamic_epsilon_exact(stats, eps, expected):
    assert bp.dynamic_epsilon(stats, eps) == pytest.approx(expected)


def test_epsilon_greedy_exploits_best_mean(monkeypatch):
    monkeypatch.setattr(bp.random, "random", lambda: 0.99)
    stats = {"a": {"mean": 0.4, "count": 10}, "b": {"mean": 0.7, "count": 10}}
    assert bp.choose_epsilon_greedy(["a", "b", "c"], stats, 0.1) == "b"
    assert bp.choose_epsilon_greedy(["c", "d"], {}, 0.0) in {"c", "d"}


def test_epsilon_greedy_explores_least_known(monkeypatch):
    monkeypatch.setattr(bp.random, "random", lambda: 0.0)
    stats = {
        "a": {"mean": 0.9, "count": 20, "var": 0.0},  # 1/21 = 0,048
        "b": {"mean": 0.1, "count": 3, "var": 0.0},  # 1/4 = 0,25
        "c": {"mean": 0.1, "count": 9, "var": 0.4},  # 1/10 + 0,2 = 0,30
    }
    assert bp.choose_epsilon_greedy(["a", "b", "c"], stats, 0.1) == "c"
    stats["c"]["var"] = 0.2  # 0,1 + 0,1 = 0,20 < 0,25
    assert bp.choose_epsilon_greedy(["a", "b", "c"], stats, 0.1) == "b"


def test_epsilon_greedy_threshold_uses_dynamic_epsilon(monkeypatch):
    # eps dinâmico = 0,1 + 0,1 (braço com 2 amostras) = 0,2: sorteio 0,15 explora, 0,25 explota.
    stats = {"a": {"mean": 0.9, "count": 2}, "b": {"mean": 0.1, "count": 50}}
    monkeypatch.setattr(bp.random, "random", lambda: 0.15)
    assert bp.choose_epsilon_greedy(["a", "b"], stats, 0.1) == "a"  # 1/3 > 1/51
    stats["b"]["count"] = 0
    assert bp.choose_epsilon_greedy(["a", "b"], stats, 0.1) == "b"  # 1/1 > 1/3
    monkeypatch.setattr(bp.random, "random", lambda: 0.25)
    assert bp.choose_epsilon_greedy(["a", "b"], stats, 0.1) == "a"  # maior média


def _ucb_reference(models, stats):
    total = sum(s.get("count", 0) for s in stats.values()) or 1
    best, best_score = None, -math.inf
    for m in models:
        s = stats.get(m, {})
        n = s.get("count", 0)
        score = math.inf if n <= 0 else s.get("mean", 0.0) + 1.4 * math.sqrt(math.log(total + 1.0) / n)
        if score > best_score:
            best, best_score = m, score
    return best


@given(
    arms=st.lists(
        st.tuples(st.floats(0.0, 1.0), st.integers(1, 500)),
        min_size=2,
        max_size=6,
    )
)
def test_ucb1_matches_reference_formula(arms):
    models = [f"m{i}" for i in range(len(arms))]
    stats = {m: {"mean": mean, "count": n} for m, (mean, n) in zip(models, arms)}
    assert bp.choose_ucb1(models, stats) == _ucb_reference(models, stats)


def test_ucb1_exploration_constant():
    # Com c = 1,4 o bônus de "b" (25 amostras) não compensa a média de "a"; com c maior, compensaria.
    stats = {"a": {"mean": 0.9, "count": 100}, "b": {"mean": 0.58, "count": 25}}
    assert bp.choose_ucb1(["a", "b"], stats) == "a"
    stats["b"]["mean"] = 0.6
    assert bp.choose_ucb1(["a", "b"], stats) == "b"


def test_ucb1_without_counts_keeps_order():
    assert bp.choose_ucb1(["x", "y"], {}) == "x"
    assert bp.choose_ucb1(["x", "y"], {"x": {"count": 4, "mean": 1.0}}) == "y"  # não visitado vence


def test_thompson_samples_beta_per_arm(monkeypatch):
    calls = []

    def fake_beta(alpha, beta):
        calls.append((alpha, beta))
        return {(3.0, 1.0): 0.9, (1.0, 1.0): 0.5}.get((alpha, beta), 0.1)

    monkeypatch.setattr(bp.np.random, "beta", fake_beta)
    stats = {
        "good": {"alpha": 3.0, "beta": 1.0},
        "bad_alpha": {"alpha": 0.0, "beta": 2.0},
        "nan": {"alpha": float("nan"), "beta": 1.0},
        "neg_beta": {"alpha": 2.0, "beta": -1.0},
    }
    assert bp.choose_thompson(["bad_alpha", "good", "nan", "neg_beta", "new"], stats) == "good"
    assert calls == [(1.0, 1.0), (3.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]


def test_thompson_prefers_strong_posterior():
    np.random.seed(0)
    stats = {"strong": {"alpha": 200.0, "beta": 2.0}, "weak": {"alpha": 2.0, "beta": 200.0}}
    picks = [bp.choose_thompson(["weak", "strong"], stats) for _ in range(20)]
    assert picks == ["strong"] * 20


@pytest.mark.parametrize(
    ("choices", "preferred", "expected"),
    [
        (("a", "a", "b"), "thompson", "a"),  # maioria vence a preferência
        (("b", "a", "a"), "epsilon_greedy", "a"),
        (("a", "b", "c"), "epsilon_greedy", "a"),
        (("a", "b", "c"), "ucb1", "b"),
        (("a", "b", "c"), "thompson", "c"),
        (("a", "b", "c"), "desconhecida", "b"),  # padrão: UCB1
    ],
)
def test_meta_combine_votes_then_preference(monkeypatch, choices, preferred, expected):
    eps, ucb, ts = choices
    monkeypatch.setattr(bp, "choose_epsilon_greedy", lambda models, stats, e: eps)
    monkeypatch.setattr(bp, "choose_ucb1", lambda models, stats: ucb)
    monkeypatch.setattr(bp, "choose_thompson", lambda models, stats: ts)
    chosen, detail = bp.meta_combine_choices(["a", "b", "c"], {}, 0.1, preferred)
    assert chosen == expected
    assert detail == {"epsilon_greedy": eps, "ucb1": ucb, "thompson": ts}


def test_meta_combine_requires_models():
    with pytest.raises(RuntimeError):
        bp.meta_combine_choices([], {}, 0.1, "ucb1")
