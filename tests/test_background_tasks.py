# Objective: Test coverage for tracked background tasks.
"""utils.background.spawn/drain and the dependency-injected spawner."""

import asyncio
from types import SimpleNamespace

import pytest

from app.utils import background


@pytest.fixture(autouse=True)
def _clean_registry():
    background._tasks.clear()
    background._daemons.clear()
    background._inflight.clear()
    yield
    background._tasks.clear()
    background._daemons.clear()
    background._inflight.clear()


class _Counter:
    """Minimal stand-in for a labelled Prometheus counter."""

    def __init__(self):
        self.counts = {}

    def labels(self, name):
        self.counts.setdefault(name, 0)
        return SimpleNamespace(inc=lambda: self.counts.__setitem__(name, self.counts[name] + 1))


@pytest.fixture
def metrics(monkeypatch):
    dropped, errors = _Counter(), _Counter()
    monkeypatch.setattr(background, "_metrics", lambda: (dropped, errors))
    return SimpleNamespace(dropped=dropped, errors=errors)


@pytest.mark.asyncio
async def test_spawn_keeps_reference_until_done():
    gate = asyncio.Event()

    async def work():
        await gate.wait()
        return 42

    task = background.spawn(work(), name="w")
    assert task in background._tasks and background.pending() == 1
    gate.set()
    assert await task == 42
    await asyncio.sleep(0)  # done-callback
    assert background.pending() == 0 and background._inflight["w"] == 0


@pytest.mark.asyncio
async def test_failures_are_counted_not_lost(metrics):
    async def boom():
        raise ValueError("x")

    task = background.spawn(boom(), name="b")
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert metrics.errors.counts == {"b": 1}


@pytest.mark.asyncio
async def test_limit_drops_and_closes_extra_work(metrics):
    gate = asyncio.Event()
    started = []

    async def work(i):
        started.append(i)
        await gate.wait()

    first = background.spawn(work(1), name="lim", limit=1)
    extra = work(2)
    assert background.spawn(extra, name="lim", limit=1) is None
    assert extra.cr_frame is None  # corrotina descartada foi fechada (sem warning de "never awaited")
    assert metrics.dropped.counts == {"lim": 1}
    gate.set()
    await first
    assert started == [1]


def test_spawn_without_loop_raises_and_closes():
    async def work():
        return None

    coro = work()
    with pytest.raises(RuntimeError):
        background.spawn(coro, name="noloop")
    assert coro.cr_frame is None


@pytest.mark.asyncio
async def test_drain_cancels_daemons_and_waits_for_others():
    finished = []

    async def forever():
        while True:
            await asyncio.sleep(3600)

    async def short():
        await asyncio.sleep(0.01)
        finished.append("short")

    async def slow():
        await asyncio.sleep(3600)

    daemon = background.spawn(forever(), name="loop", daemon=True)
    background.spawn(short(), name="short")
    stuck = background.spawn(slow(), name="slow")
    await background.drain(timeout=0.2)
    assert daemon.cancelled() and stuck.cancelled()
    assert finished == ["short"]


@pytest.mark.asyncio
async def test_spawn_via_deps_prefers_injected_spawner():
    from app.services.router_services import spawn_via_deps

    calls = []

    async def work():
        return None

    deps = {"spawn_background": lambda coro, **kw: calls.append(kw) or coro.close()}
    spawn_via_deps(deps, work(), name="n", limit=3)
    assert calls == [{"name": "n", "limit": 3}]

    created = []
    legacy = {"asyncio": SimpleNamespace(create_task=lambda coro: created.append(coro) or coro.close())}
    spawn_via_deps(legacy, work(), name="n")
    assert len(created) == 1
