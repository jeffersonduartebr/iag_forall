# Objective: Test coverage for Redis-backed distributed coordination (fakeredis).
"""Global semaphore, sliding-window limit and idempotency against real Redis semantics."""

from types import SimpleNamespace

import pytest
from app.utils import redis_distributed as rd


def _settings(env="development", required="1"):
    values = {"ENV": env, "REDIS_REQUIRED_IN_PRODUCTION": required}
    return SimpleNamespace(get=lambda key, default=None: values.get(key, default))


@pytest.fixture
def no_redis(monkeypatch):
    async def _none():
        return None

    monkeypatch.setattr("app.utils.redis_async_ops.get_redis_async", _none)


@pytest.mark.parametrize(
    ("env", "flag", "expected"),
    [("production", "1", True), ("production", "off", False), ("development", "1", False)],
)
def test_redis_required_only_in_production(monkeypatch, env, flag, expected):
    monkeypatch.setattr(rd, "settings", _settings(env, flag))
    assert rd._redis_required() is expected


@pytest.mark.asyncio
async def test_semaphore_caps_holders_and_releases(fake_aioredis):
    first, second, third = (rd.RedisGlobalSemaphore("gpu", 2) for _ in range(3))
    assert await first.acquire() is True
    assert await second.acquire() is True
    assert await third.acquire() is False
    # O excedente é devolvido: o contador reflete só os dois detentores.
    assert int(await fake_aioredis.get("bp:sem:gpu")) == 2
    assert 0 < await fake_aioredis.ttl("bp:sem:gpu") <= 120

    await first.release()
    await first.release()  # segunda liberação é no-op
    assert int(await fake_aioredis.get("bp:sem:gpu")) == 1
    assert await third.acquire() is True


@pytest.mark.asyncio
async def test_semaphore_release_without_acquire_is_noop(fake_aioredis):
    await rd.RedisGlobalSemaphore("idle", 1).release()
    assert await fake_aioredis.get("bp:sem:idle") is None


@pytest.mark.asyncio
async def test_semaphore_fails_open_without_redis(no_redis):
    sem = rd.RedisGlobalSemaphore("any", 1)
    assert await sem.acquire() is True
    assert sem._acquired is False  # nada a liberar


@pytest.mark.asyncio
async def test_semaphore_fails_open_on_redis_error(monkeypatch):
    async def _boom(build):
        raise ConnectionError("down")

    monkeypatch.setattr("app.utils.redis_async_ops.redis_pipeline_execute", _boom)
    assert await rd.RedisGlobalSemaphore("any", 1).acquire() is True


@pytest.mark.asyncio
async def test_sliding_window_limit_counts_requests(fake_aioredis):
    over = [await rd.redis_sliding_window_limit("t1", max_requests=3, window_seconds=60) for _ in range(5)]
    # A contagem é lida antes de registrar a requisição atual.
    assert over == [False, False, False, True, True]
    # Rejeitadas não ocupam a janela (antes: 5, e o tenant nunca saía do 429 sob carga).
    assert await fake_aioredis.zcard("tenant-rl:t1") == 3
    assert await rd.redis_sliding_window_limit("t2", max_requests=3, window_seconds=60) is False


@pytest.mark.asyncio
async def test_sliding_window_counts_requests_sharing_a_timestamp(fake_aioredis, monkeypatch):
    monkeypatch.setattr(rd.time, "time", lambda: 1000.0)  # todas no mesmo instante
    over = [await rd.redis_sliding_window_limit("t3", max_requests=2, window_seconds=60) for _ in range(3)]
    assert over == [False, False, True]  # com membro str(now) as três eram uma só


@pytest.mark.asyncio
async def test_sliding_window_limit_without_redis(monkeypatch, no_redis):
    monkeypatch.setattr(rd, "settings", _settings("development"))
    assert await rd.redis_sliding_window_limit("t", max_requests=1, window_seconds=1) is False

    monkeypatch.setattr(rd, "settings", _settings("production"))
    with pytest.raises(RuntimeError):
        await rd.redis_sliding_window_limit("t", max_requests=1, window_seconds=1)


@pytest.mark.asyncio
async def test_sliding_window_limit_error_policy(monkeypatch):
    async def _boom(build):
        raise ConnectionError("down")

    monkeypatch.setattr("app.utils.redis_async_ops.redis_pipeline_execute", _boom)
    monkeypatch.setattr(rd, "settings", _settings("development"))
    assert await rd.redis_sliding_window_limit("t", max_requests=1, window_seconds=1) is False

    monkeypatch.setattr(rd, "settings", _settings("production"))
    with pytest.raises(ConnectionError):
        await rd.redis_sliding_window_limit("t", max_requests=1, window_seconds=1)


@pytest.mark.asyncio
async def test_idempotency_roundtrip(fake_aioredis):
    assert await rd.redis_idempotency_get("k") is None
    await rd.redis_idempotency_set("k", {"answer": "ok", "n": 1}, ttl_s=5)
    assert await rd.redis_idempotency_get("k") == {"answer": "ok", "n": 1}
    assert 5 < await fake_aioredis.ttl("idemp:k") <= 30  # TTL mínimo de 30 s

    await fake_aioredis.set("idemp:bad", b"{not json")
    assert await rd.redis_idempotency_get("bad") is None


def test_idempotency_key_is_stable_and_scoped():
    key = rd.compute_idempotency_key(tenant_id="a", query="q", modality="text")
    assert key == rd.compute_idempotency_key(tenant_id="a", query="q", modality="text")
    assert len(key) == 32
    assert key != rd.compute_idempotency_key(tenant_id="b", query="q", modality="text")
    assert rd.compute_idempotency_key(tenant_id=None, query="q", modality="text") == rd.compute_idempotency_key(
        tenant_id="anon", query="q", modality="text"
    )
