# Objective: A populated decision record must still produce a valid /query response (regression of PR #24).
from __future__ import annotations

from types import SimpleNamespace

from app.services.query_response_builder import build_query_response
from app.services.route_decision import Candidate, decision_record
from app.services.router_provider_stage import build_result
from app.services.router_stages import RouteChoice


def test_route_pareto_front_matches_the_response_contract():
    cands = [
        Candidate(model="a", quality=8, latency_s=2, cost_usd=0.01, risk=0.1, score=0.9),
        Candidate(model="b", quality=6, latency_s=1, cost_usd=0.001, risk=0.2, score=0.7),
    ]
    decisao = decision_record(cands, chosen="a", top2=["a", "b"], weights={"w_quality": 1.0}, uncertainty=0.2)
    choice = RouteChoice(chosen="a", top2=["a", "b"], decision=decisao)
    deps = {
        "parse_meta_cost": lambda **kw: (10, 20, 0.01, 0.0, {"prompt_tokens": 10, "completion_tokens": 20}),
        "get_model_cost": lambda *a: 0.0,
        "logger": SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None),
        "ROUTER_ROUTE_COST": SimpleNamespace(labels=lambda **k: SimpleNamespace(inc=lambda v: None)),
        "ROUTER_ATTEMPTS_PER_QUERY": SimpleNamespace(observe=lambda v: None),
    }
    ctx = SimpleNamespace(
        deps=deps,
        modality="text",
        image_b64=None,
        start_time=0.0,
        stage_timings_ms={},
        observe_stage=lambda *a: None,
        tenant_id="t",
        hints={},
    )
    outcome = SimpleNamespace(
        meta={}, out="resposta", chosen="a", fallback_used=False, models_tried=["a"], errors=[], retry_count=0
    )
    result = build_result(ctx, choice, outcome, 0.2, {})
    resposta = build_query_response(result, "cid")
    frente = resposta.diagnostics.route.pareto_front
    assert {c["model"] for c in frente} == set(decisao["pareto_front"])
