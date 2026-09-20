# Objective: A routing decision must be reconstructible from what was stored.
"""``candidates`` and ``pareto_front`` were written as literal empty lists.

Both routing paths emitted them empty, so the frente de Pareto that names the
central contribution of this work never reached disk. The stored data answered
*which* model replied and never *why* — and the inputs to that decision (the
Thompson samples, the shared EMAs, the NSGA-II weights in force) are all gone
by the time the row is written, so no later analysis can recover them.

These tests pin the two properties that make the record worth storing: the
front is computed over the raw objectives rather than the scalarised score, and
the weights travel with it.
"""

import pytest
from app.services.route_decision import Candidate, decision_record, pareto_front


def c(model, quality, latency, cost, *, risk=1.0, score=0.0):
    return Candidate(model=model, quality=quality, latency_s=latency, cost_usd=cost, risk=risk, score=score)


# ---------------------------------------------------------------------------
# The front itself
# ---------------------------------------------------------------------------


def test_a_dominated_candidate_is_excluded():
    """Worse on every objective at once, so nothing can justify it."""
    best = c("bom", quality=9.0, latency=1.0, cost=0.001)
    worse = c("pior", quality=5.0, latency=3.0, cost=0.010)
    assert [x.model for x in pareto_front([best, worse])] == ["bom"]


def test_a_trade_off_keeps_both():
    """Slow and excellent versus fast and mediocre: neither dominates."""
    excelente = c("lento", quality=9.5, latency=8.0, cost=0.020)
    rapido = c("rapido", quality=6.0, latency=0.4, cost=0.001)
    assert {x.model for x in pareto_front([excelente, rapido])} == {"lento", "rapido"}


def test_an_equal_candidate_is_not_dominated_by_its_twin():
    """Domination needs strict superiority on at least one objective."""
    a = c("a", quality=7.0, latency=2.0, cost=0.005)
    b = c("b", quality=7.0, latency=2.0, cost=0.005)
    assert len(pareto_front([a, b])) == 2


def test_being_equal_everywhere_but_worse_on_one_is_domination():
    a = c("a", quality=7.0, latency=2.0, cost=0.005)
    b = c("b", quality=7.0, latency=2.0, cost=0.009)
    assert [x.model for x in pareto_front([a, b])] == ["a"]


def test_an_empty_roster_has_an_empty_front():
    assert pareto_front([]) == []


def test_the_order_given_is_preserved():
    """The record is read by humans; a stable order makes rows comparable."""
    # Mais qualidade custa mais latência: nenhum domina outro.
    models = [c(f"m{i}", quality=6.0 + i, latency=float(i), cost=0.0) for i in range(4)]
    assert [x.model for x in pareto_front(models)] == [m.model for m in models]


# ---------------------------------------------------------------------------
# What the front is computed over
# ---------------------------------------------------------------------------


def test_the_front_ignores_the_scalarised_score():
    """This is the whole point of storing the front separately.

    A weighted sum collapses three objectives onto one line and can only ever
    return a single point of the front. A record that stored only the winner of
    that sum could not distinguish "the best trade-off under today's weights"
    from "the only option that was not dominated".
    """
    vencedor = c("vencedor", quality=6.0, latency=0.5, cost=0.001, score=100.0)
    outro = c("outro", quality=9.9, latency=9.0, cost=0.050, score=-50.0)
    front = {x.model for x in pareto_front([vencedor, outro])}
    assert front == {"vencedor", "outro"}, "o score escalarizado contaminou a frente"


# ---------------------------------------------------------------------------
# The stored record
# ---------------------------------------------------------------------------


@pytest.fixture
def record():
    candidates = [
        c("ollama/phi4", quality=6.0, latency=0.8, cost=0.0, score=5.2),
        c("openai/gpt-4o", quality=9.0, latency=2.5, cost=0.03, score=4.1),
        c("openai/caro-e-mau", quality=4.0, latency=5.0, cost=0.09, score=-2.0),
    ]
    return decision_record(
        candidates,
        chosen="ollama/phi4",
        top2=["ollama/phi4", "openai/gpt-4o"],
        weights={"w_quality": 1.0, "w_latency": 0.5, "w_cost": 50.0},
        uncertainty=0.42,
    )


def test_every_candidate_is_recorded_with_its_objectives(record):
    assert len(record["candidates"]) == 3
    phi = next(x for x in record["candidates"] if x["model"] == "ollama/phi4")
    assert phi["quality"] == 6.0 and phi["latency_s"] == 0.8 and phi["cost_usd"] == 0.0
    assert phi["risk"] == 1.0 and phi["score"] == 5.2


def test_the_dominated_candidate_is_marked_off_the_front(record):
    on_front = {x["model"] for x in record["candidates"] if x["on_pareto_front"]}
    assert "openai/caro-e-mau" not in on_front
    assert on_front == set(record["pareto_front"])


def test_the_weights_travel_with_the_record(record):
    """A background updater rewrites them, so a score stored without the
    weights that produced it cannot be reinterpreted afterwards."""
    assert record["weights"] == {"w_quality": 1.0, "w_latency": 0.5, "w_cost": 50.0}


def test_the_chosen_model_and_uncertainty_are_recorded(record):
    assert record["chosen"] == "ollama/phi4"
    assert record["top2"] == ["ollama/phi4", "openai/gpt-4o"]
    assert record["uncertainty"] == 0.42


def test_the_record_is_json_serialisable(record):
    """It is stored as a LONGTEXT column, so anything exotic would be lost."""
    import json

    assert json.loads(json.dumps(record)) == record


def test_missing_weights_do_not_break_the_record():
    """The NSGA updater may not have run yet on a fresh deployment."""
    out = decision_record([c("m", 1.0, 1.0, 1.0)], chosen="m", top2=["m"], weights={}, uncertainty=0.0)
    assert out["weights"] == {}
    assert out["pareto_front"] == ["m"]


# ---------------------------------------------------------------------------
# It has to survive the whole way to the row
# ---------------------------------------------------------------------------


def test_the_scorer_exposes_the_detail_it_used_to_discard():
    """``choose_top2_models`` computed all of this and returned two names."""
    from app.router_strategy import choose_top2_models, score_candidates

    assert callable(score_candidates)
    # A fachada continua a existir com o contrato antigo, porque o pipeline, os
    # testes e o dicionário de dependências a tratam por este nome.
    assert callable(choose_top2_models)


def test_the_response_payload_carries_the_decision():
    """The two fields that were literal empty lists."""
    import inspect

    from app.services import router_provider_stage

    source = inspect.getsource(router_provider_stage)
    assert '"pareto_front": choice.decision.get("pareto_front", [])' in source
    assert '"candidates": choice.decision.get("candidates", [])' in source
    assert '"pareto_front": []' not in source


def test_the_feedback_payload_carries_the_decision_and_the_correlation_id():
    from app.services.feedback_payload import build_feedback_payload

    result = {
        "metadata": {"correlation_id": "abc-123", "uncertainty_score": 0.3},
        "decision": {"chosen": "m", "pareto_front": ["m"]},
    }
    payload = build_feedback_payload(result, tenant_id="acme", include_raw=False)
    assert payload["decision"] == {"chosen": "m", "pareto_front": ["m"]}
    assert payload["correlation_id"] == "abc-123"


def test_the_insert_accepts_the_two_new_columns():
    import inspect

    from app.query_service import insert_query_log

    params = inspect.signature(insert_query_log).parameters
    for name in ("decision", "correlation_id"):
        assert name in params, name
        assert params[name].default is None


def test_persist_log_passes_the_decision_through():
    from app.services.feedback_stages import FeedbackRequest, Quality, _formative_fields

    fb = FeedbackRequest(
        query="q", answer="a", chosen_model="m", modality="text", latency_s=1.0, cost_val=0.0,
        raw_payload={"decision": {"chosen": "m"}, "correlation_id": "xyz"},
    )
    fields = _formative_fields(Quality(6.0, "bandit_proxy"), fb)
    assert fields["decision"] == {"chosen": "m"}
    assert fields["correlation_id"] == "xyz"


def test_a_cache_hit_records_no_decision():
    """Honest emptiness: a cache hit made no comparison, so there is nothing
    to record. This is the one place an empty list is the truth."""
    from app.services.feedback_stages import FeedbackRequest, Quality, _formative_fields

    fb = FeedbackRequest(query="q", answer="a", chosen_model="semantic_cache",
                         modality="text", latency_s=0.0, cost_val=0.0, raw_payload={"decision": {}})
    assert _formative_fields(Quality(6.0, "bandit_proxy"), fb)["decision"] is None
