# Objective: The error budget is per provider: one failing upstream leaves the pool, the others keep serving.
from __future__ import annotations

import pytest
from app.services import provider_budget as pb

_POLICY = {"ERROR_BUDGET_WINDOW_S": 30, "ERROR_BUDGET_THRESHOLD": 0.2, "ERROR_BUDGET_MIN_REQUESTS": 5}


def _getter(key, default=None):
    return _POLICY.get(key, default)


@pytest.mark.asyncio
async def test_only_providers_past_threshold_with_enough_samples_are_over(fake_aioredis):
    for ok in [False] * 3 + [True] * 7:
        await pb._record(_getter, "ollama", ok)  # 30% de falha, 10 amostras
    for _ in range(4):
        await pb._record(_getter, "gemini", False)  # 100% de falha, mas só 4 amostras
    for _ in range(6):
        await pb._record(_getter, "openrouter", True)
    assert await pb.providers_over_budget(_getter) == {"ollama"}


@pytest.mark.asyncio
async def test_disabled_budget_and_redis_failures_exclude_nobody(monkeypatch):
    assert await pb.providers_over_budget(lambda k, d=None: "0" if k == "ERROR_BUDGET_ENABLED" else d) == set()

    async def boom(key):
        raise ConnectionError("redis fora")

    monkeypatch.setattr(pb, "redis_hgetall_map", boom)
    assert await pb.providers_over_budget(_getter) == set()


def test_provider_prefix_and_counter_parsing():
    assert pb.provider_of("openrouter/deepseek/x") == "openrouter" and pb.provider_of("") == ""
    totals = pb._sum_counters([{"ollama:total": "2", "ollama:errors": "1"}, {"ollama:total": "3"}])
    assert totals == {"ollama": {"total": 5, "errors": 1}}


@pytest.mark.asyncio
async def test_concurrent_async_redis_connects_do_not_deadlock_the_event_loop(monkeypatch):
    """The client lock (threading) was held across ``await ping()``: a second coroutine connecting at the
    same time blocked the loop thread forever (found when provider samples were recorded concurrently)."""
    import asyncio

    import redis.asyncio as aioredis
    from app.utils import redis_client as rc

    pings = []

    class _Unreachable:  # Redis fora: o 1º ping demora (conexão), os seguintes falham na hora
        def __init__(self, **kwargs):
            pass

        async def ping(self):
            pings.append(1)
            if len(pings) == 1:
                await asyncio.sleep(0.05)
            raise ConnectionError("redis fora")

    monkeypatch.setattr(aioredis, "Redis", _Unreachable)
    monkeypatch.setattr(rc, "_async_redis_client", None)
    clients = await asyncio.wait_for(asyncio.gather(rc.get_redis_async(), rc.get_redis_async()), timeout=5)
    assert clients == [None, None]
