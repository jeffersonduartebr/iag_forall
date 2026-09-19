# Objective: Property-based tests (hypothesis) for bandit policies, retrieval fusion and settings coercion.
"""Invariants of the routing and retrieval helpers, checked over generated inputs."""

import statistics
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from app.config import settings_types
from app.services import bandit_policy
from app.services.retrieval_assembly import _trim_context_to_budget, reciprocal_rank_fusion
from hypothesis import given
from hypothesis import strategies as st

from app import bandits, semantic_cache

model_names = st.lists(st.text(alphabet="abcdefgh/:", min_size=1, max_size=8), min_size=1, max_size=8, unique=True)
arm_stats = st.fixed_dictionaries(
    {
        "mean": st.floats(0.0, 1.0),
        "count": st.integers(0, 1000),
        "var": st.floats(0.0, 1.0),
        "alpha": st.floats(0.01, 100.0),
        "beta": st.floats(0.01, 100.0),
    }
)


@st.composite
def arms(draw):
    models = draw(model_names)
    stats = {m: draw(arm_stats) for m in models if draw(st.booleans())}
    return models, stats


# ---------------------------------------------------------------- bandit


@given(case=arms(), eps=st.floats(0.0, 1.0))
def test_dynamic_epsilon_in_unit_interval(case, eps):
    _, stats = case
    assert 0.0 <= bandit_policy.dynamic_epsilon(stats, eps) <= 1.0


@given(case=arms())
def test_ucb1_prefers_unvisited_arms(case):
    models, stats = case
    chosen = bandit_policy.choose_ucb1(models, stats)
    assert chosen in models
    unvisited = [m for m in models if stats.get(m, {}).get("count", 0) <= 0]
    if unvisited:
        assert chosen in unvisited


@given(case=arms(), eps=st.floats(0.0, 1.0), strategy=st.sampled_from(["ucb1", "thompson", "epsilon_greedy"]))
def test_meta_choice_is_a_candidate(case, eps, strategy):
    models, stats = case
    chosen, votes = bandit_policy.meta_combine_choices(models, stats, eps, strategy)
    assert chosen in models


@given(rewards=st.lists(st.floats(0.0, 1.0), min_size=1, max_size=40))
def test_welford_update_matches_batch_statistics(rewards):
    store = {}
    with (
        patch.object(bandits, "centroids_online_update", lambda query: None),
        patch.object(bandits, "_auto_context_labels", lambda query, modality: ["ctx"]),
        patch.object(bandits, "_get_ctx_stats", lambda ctx: dict(store.get(ctx, {}))),
        patch.object(bandits, "_set_ctx_stats", lambda ctx, stats: store.__setitem__(ctx, stats)),
        patch.object(bandits, "_batch_upsert_ctx_db", lambda updates: None),
    ):
        for r in rewards:
            bandits.bandit_update("m", "q", r)
    final = store["ctx"]["m"]
    assert final["count"] == len(rewards)
    assert final["mean"] == pytest.approx(statistics.fmean(rewards), abs=1e-9)
    expected_var = statistics.variance(rewards) if len(rewards) > 1 else 0.0
    assert final["var"] == pytest.approx(expected_var, abs=1e-9)
    assert final["alpha"] == pytest.approx(1.0 + sum(rewards))
    assert final["beta"] == pytest.approx(1.0 + len(rewards) - sum(rewards))


# ---------------------------------------------------------------- recuperação


doc_ids = st.lists(st.sampled_from([f"d{i}" for i in range(12)]), max_size=12, unique=True)


@given(dense=doc_ids, sparse=doc_ids)
def test_rrf_is_union_without_duplicates(dense, sparse):
    fused = reciprocal_rank_fusion(dense, sparse)
    assert len(fused) == len(set(fused))
    assert set(fused) == set(dense) | set(sparse)


@given(dense=doc_ids, sparse=doc_ids)
def test_rrf_keeps_a_doc_ranked_first_by_both(dense, sparse):
    if dense and sparse and dense[0] == sparse[0]:
        assert reciprocal_rank_fusion(dense, sparse)[0] == dense[0]


@given(docs=st.lists(st.text(max_size=200), max_size=15), budget=st.integers(-5, 200))
def test_trim_never_exceeds_budget_and_preserves_order(docs, budget):
    trimmed = _trim_context_to_budget(docs, budget)
    if budget <= 0:
        assert trimmed == docs
        return
    assert sum(len(d) for d in trimmed) <= budget * 4
    source = iter(d for d in docs if d)
    for piece in trimmed:  # cada trecho é prefixo do próximo documento não vazio, na ordem
        assert any(doc.startswith(piece) for doc in source)


@given(query=st.text(max_size=200))
def test_query_normalization_is_idempotent(query):
    enabled = SimpleNamespace(get=lambda key, default=None: "1")
    with patch.object(semantic_cache, "settings", enabled):
        once = semantic_cache._normalize_query(query)
        assert semantic_cache._normalize_query(once) == once
        assert "  " not in once and once == once.strip()


# ---------------------------------------------------------------- settings

anything = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=40),
    st.binary(max_size=20),
    st.lists(st.integers(), max_size=3),
)


@given(value=anything)
def test_settings_coercion_never_raises(value):
    assert isinstance(settings_types.as_bool(value), bool)
    assert settings_types.as_bool(settings_types.as_bool(value)) == settings_types.as_bool(value)
    assert isinstance(settings_types.as_int(value, 7), int)
    assert isinstance(settings_types.as_float(value, 0.5), float)
    assert isinstance(settings_types.as_list(None if value is None else str(value)), list)


@given(items=st.lists(st.text(alphabet="abcxyz-_.", min_size=1, max_size=10), max_size=6))
def test_as_list_roundtrips_comma_separated(items):
    assert settings_types.as_list(",".join(items)) == items
