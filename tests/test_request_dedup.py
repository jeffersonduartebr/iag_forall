# Objective: Regression tests for in-flight request deduplication (concurrency and failure paths).
"""RequestDeduplicator: followers share the leader's result without serializing other requests."""

import asyncio
import gc

import pytest
from app.reliability import RequestDeduplicator


@pytest.fixture
def dedup():
    RequestDeduplicator._instance = None
    d = RequestDeduplicator()
    yield d
    RequestDeduplicator._instance = None


@pytest.mark.asyncio
async def test_follower_shares_leader_result_with_single_execution(dedup):
    release, calls = asyncio.Event(), []

    async def slow():
        calls.append(1)
        await release.wait()
        return "resposta"

    leader = asyncio.create_task(dedup.deduplicate("q", "m", slow))
    await asyncio.sleep(0)
    follower = asyncio.create_task(dedup.deduplicate("q", "m", slow))
    await asyncio.sleep(0)
    release.set()
    assert await asyncio.gather(leader, follower) == ["resposta", "resposta"]
    assert calls == [1]
    assert dedup._in_flight == {}


@pytest.mark.asyncio
async def test_waiting_follower_does_not_block_other_requests(dedup):
    release = asyncio.Event()

    async def slow():
        await release.wait()
        return "lento"

    async def fast():
        return "rápido"

    leader = asyncio.create_task(dedup.deduplicate("q", "m", slow))
    await asyncio.sleep(0)
    follower = asyncio.create_task(dedup.deduplicate("q", "m", slow))
    await asyncio.sleep(0)
    # Antes, o seguidor esperava segurando o lock e esta chamada ficava presa até o LLM responder.
    assert await asyncio.wait_for(dedup.deduplicate("outra", "m", fast), timeout=1.0) == "rápido"
    release.set()
    await asyncio.gather(leader, follower)


@pytest.mark.asyncio
async def test_failed_leader_without_followers_leaves_no_unretrieved_exception(dedup):
    errors = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, ctx: errors.append(ctx.get("message")))

    async def boom():
        raise RuntimeError("provider timeout")

    with pytest.raises(RuntimeError):
        await dedup.deduplicate("q", "m", boom)
    gc.collect()
    await asyncio.sleep(0)
    loop.set_exception_handler(None)
    assert errors == []
    assert dedup._in_flight == {}


@pytest.mark.asyncio
async def test_followers_recover_when_leader_fails_or_is_cancelled(dedup):
    release = asyncio.Event()

    async def hang():
        await release.wait()

    async def ok():
        return "próprio"

    leader = asyncio.create_task(dedup.deduplicate("q", "m", hang))
    await asyncio.sleep(0)
    follower = asyncio.create_task(dedup.deduplicate("q", "m", ok))
    await asyncio.sleep(0)
    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader
    # O seguidor não fica preso: executa por conta própria.
    assert await asyncio.wait_for(follower, timeout=1.0) == "próprio"
    assert dedup._in_flight == {}
