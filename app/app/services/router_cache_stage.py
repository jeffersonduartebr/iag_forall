# Objective: The cache stage of the routing pipeline — lookup, breaker, hit payload.
"""Semantic-cache stage, split out of ``router_stages``.

Two things live here and nowhere else: the circuit breaker that protects the
cache lookup, and the response bundle a hit produces. Both were extracted so
``router_stages`` could go back under its size ceiling — and because the hit
payload is where the routing decision is recorded for a request that never
reached a provider.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from app.utils.breaker_async import call_through_breaker

from .router_stages import RouteContext, _record_breakers, quietly


async def _lookup_cache(ctx: RouteContext) -> Optional[Dict[str, Any]]:
    deps = ctx.deps
    breaker = deps["_dep_cache_breaker"]
    kwargs = {"modality": ctx.modality, "image_b64": ctx.image_b64, "tenant_id": ctx.tenant_id}
    if breaker.current_state == "open":
        return None
    try:
        return await call_through_breaker(breaker, deps["check_cache"], ctx.query, **kwargs)
    except Exception:
        # Sem cache segue-se em frente; repetir a chamada sem o breaker, como
        # se fazia aqui, anulava o próprio breaker — e com `call_async` (que é
        # Tornado-only e levanta NameError antes sequer de chamar `check_cache`)
        # esse ramo corria SEMPRE, deixando o breaker permanentemente inerte.
        return None


def _cache_hit_result(ctx: RouteContext, cached: Dict[str, Any]) -> Dict[str, Any]:
    deps = ctx.deps
    deps["logger"].info(f"[router] Cache HIT ({cached.get('similarity', 0):.2f})")
    quietly(lambda: deps["ROUTER_ROUTE_COST"].labels(route_type="cache").inc(0.0))
    quietly(lambda: deps["ROUTER_ATTEMPTS_PER_QUERY"].observe(1))
    return {
        "model": "semantic_cache",
        "modality": ctx.modality,
        "answer": cached.get("text", ""),
        "image_output_b64": cached.get("image_output_b64"),
        "latency_s": round(time.time() - ctx.start_time, 3),
        "estimated_cost_usd": 0.0,
        "cost_per_1k": 0.0,
        "metadata": {"cached": True, "stage_timings_ms": dict(ctx.stage_timings_ms)},
        "route": {
            "chosen_model": "semantic_cache",
            "objectives": {"latency": 0, "cost": 0, "uncertainty": 0},
            # Vazio com razão: um cache hit não fez comparação nenhuma, logo
            # não há decisão de routing para registar. Isto é diferente do
            # caminho normal, onde o vazio era informação deitada fora.
            "pareto_front": [],
            "explanation": "Cache",
            "fallback": {"used": False, "models_tried": ["semantic_cache"], "errors": []},
        },
        "candidates": [],
    }


async def try_cache_hit(ctx: RouteContext) -> Optional[Dict[str, Any]]:
    """Semantic-cache lookup behind its circuit breaker; returns the router result on a hit."""
    if not ctx.use_cache:
        return None
    started = time.time()
    cached = None
    try:
        cached = await _lookup_cache(ctx)
        _record_breakers(ctx.deps)
    except Exception:
        quietly(lambda: ctx.deps["DEPENDENCY_FAILURES"].labels(dependency="cache").inc())
        _record_breakers(ctx.deps)
    finally:
        ctx.observe_stage("cache_lookup", started)
    return _cache_hit_result(ctx, cached) if cached else None
