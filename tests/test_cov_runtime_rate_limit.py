# Objective: Adaptive limiter failure paths — Redis down/flapping, per-tenant buckets, hysteresis, 429 contract.
"""Coverage of app.middleware.rate_limit paths that matter when production degrades."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.middleware import rate_limit as rl
from starlette.requests import Request
from starlette.responses import Response

CONGESTED = {"current_limit": 1, "total_inflight": 0, "max_queue_wait_ms": 0.0, "utilization": 0.0, "pressure_state": "congested"}
ELEVATED = dict(CONGESTED, pressure_state="elevated")


class _Settings:
    def __init__(self, values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


def _request(path, headers=None, query_string=b"", client="10.0.0.1"):
    scope = {
        "type": "http", "method": "GET", "path": path, "raw_path": path.encode(), "query_string": query_string,
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": (client, 1234), "server": ("testserver", 80), "scheme": "http",
    }
    return Request(scope)


async def _ok(_request):
    return Response("ok")


@pytest.fixture
def limiter(monkeypatch, fake_clock):
    """Fresh store/tracker, hysteresis 1, memory backend (Redis down) and a controllable snapshot."""
    values = {"ADAPTIVE_LIMITER_HYSTERESIS_WINDOWS": "1", "ADAPTIVE_LIMITER_ADMIN_PER_SLOT_ELEVATED": "2"}
    monkeypatch.setattr(rl, "settings", _Settings(values))
    monkeypatch.setattr(rl, "rate_limit_store", rl.RateLimitStore())
    monkeypatch.setattr(rl, "_pressure_tracker", rl.PressureStateTracker())
    monkeypatch.setattr(rl, "get_backpressure", lambda: SimpleNamespace(get_stats=lambda: {"utilization": 0.0}))
    state = {"snapshot": ELEVATED}
    monkeypatch.setattr(rl, "get_ollama_admission_snapshot", lambda: state["snapshot"])

    async def _no_redis():
        return None

    monkeypatch.setattr(rl, "get_redis_async", _no_redis)
    return SimpleNamespace(values=values, state=state, clock=fake_clock(rl), mw=rl.RateLimitMiddleware(app=None))


def test_config_parsers_fall_back_and_clamp(monkeypatch):
    monkeypatch.setattr(rl, "settings", _Settings({"A": "abc", "B": "nope", "C": "9", "D": "YES"}))
    assert rl._as_int("A", 7, minimum=1) == 7
    assert rl._as_float("B", 0.5, minimum=0.1) == 0.5
    assert rl._as_float("C", 0.5, minimum=0.1, maximum=1.5) == 1.5
    assert rl._as_bool("D", False) is True
    assert rl._adaptive_limiter_config()["window_seconds"] == 15


def test_get_redis_returns_none_when_client_import_explodes(monkeypatch):
    def _boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr("app.utils.redis_client.get_redis_async_safe", _boom)
    assert rl._get_redis() is None
    monkeypatch.setattr("app.utils.redis_client.get_redis_async_safe", lambda: "client")
    assert rl._get_redis() == "client"


@pytest.mark.asyncio
async def test_redis_down_uses_memory_and_reprobes_only_after_interval(limiter, monkeypatch):
    store = rl.rate_limit_store
    probes = []

    async def _probe():
        probes.append(1)
        return None

    monkeypatch.setattr(rl, "get_redis_async", _probe)
    assert await store.is_rate_limited("t:a", max_requests=2, window_seconds=10) is False
    assert await store.is_rate_limited("t:a", max_requests=2, window_seconds=10) is False
    assert await store.is_rate_limited("t:a", max_requests=2, window_seconds=10) is True
    assert len(probes) == 1  # no reprobe storm while Redis is down
    limiter.clock.advance(11)  # window slides and reprobe interval elapses
    assert await store.is_rate_limited("t:a", max_requests=2, window_seconds=10) is False
    assert len(probes) == 2
    assert store.get_stats() == {"using_redis": False, "memory_entries": 1, "memory_total_requests": 1}
    assert await store.is_rate_limited("t:a", max_requests=0, window_seconds=10) is True


@pytest.mark.asyncio
async def test_redis_backend_counts_distributed_window(limiter, monkeypatch, fake_aioredis):
    async def _client():
        return fake_aioredis

    monkeypatch.setattr(rl, "get_redis_async", _client)
    store = rl.rate_limit_store
    results = []
    for _ in range(3):
        results.append(await store.is_rate_limited("t:b", max_requests=2, window_seconds=10))
        limiter.clock.advance(1)
    assert results == [False, False, True]
    assert store.get_stats()["using_redis"] is True
    assert await fake_aioredis.ttl("adaptive-limit:t:b") > 0
    limiter.clock.advance(20)
    assert await store.is_rate_limited("t:b", max_requests=2, window_seconds=10) is False


@pytest.mark.asyncio
async def test_redis_backend_same_timestamp_and_rejections_match_memory(limiter, monkeypatch, fake_aioredis):
    async def _client():
        return fake_aioredis

    monkeypatch.setattr(rl, "get_redis_async", _client)
    store = rl.rate_limit_store
    # Relógio parado: com o membro str(now) os pedidos colapsavam num só e nunca havia 429.
    results = [await store.is_rate_limited("t:s", max_requests=2, window_seconds=10) for _ in range(4)]
    assert results == [False, False, True, True]
    assert await fake_aioredis.zcard("adaptive-limit:t:s") == 2  # rejeitados não contam


@pytest.mark.asyncio
async def test_redis_pipeline_failure_falls_back_to_memory(limiter, monkeypatch):
    async def _client():
        return object()

    async def _pipeline_down(_build):
        return None

    monkeypatch.setattr(rl, "get_redis_async", _client)
    monkeypatch.setattr("app.utils.redis_async_ops.redis_pipeline_execute", _pipeline_down)
    store = rl.rate_limit_store
    assert await store.is_rate_limited("t:c", max_requests=1, window_seconds=10) is False
    assert store._use_redis is False
    assert store._next_redis_probe_at == pytest.approx(limiter.clock.now + 10)
    assert await store.is_rate_limited("t:c", max_requests=1, window_seconds=10) is True  # memory remembers


@pytest.mark.asyncio
async def test_cleanup_drops_stale_buckets_and_metric_errors_are_swallowed(limiter, monkeypatch):
    store = rl.rate_limit_store
    await store.is_rate_limited("old", max_requests=5, window_seconds=10)
    limiter.clock.advance(store.ENTRY_TTL_SECONDS + 1)
    await store.is_rate_limited("new", max_requests=5, window_seconds=10)
    monkeypatch.setattr(rl, "ADAPTIVE_LIMITER_IDENTITY_BUCKETS", None)  # .labels() raises
    assert await store.cleanup() == 1
    assert list(store._memory_store) == ["new"]


@pytest.mark.asyncio
async def test_hysteresis_ignores_flapping_candidates(monkeypatch):
    monkeypatch.setattr(rl, "ADAPTIVE_LIMITER_STATE", None)  # publish failure must not break admission
    tracker = rl.PressureStateTracker()
    assert await tracker.update("elevated", 2) == "normal"
    assert await tracker.update("congested", 2) == "normal"  # different candidate restarts the count
    assert await tracker.update("congested", 2) == "congested"
    assert await tracker.update("congested", 2) == "congested"
    assert await tracker.update("normal", 2) == "congested"


@pytest.mark.asyncio
async def test_disabled_limiter_and_polling_bypass_admission(limiter):
    limiter.values["ADAPTIVE_LIMITER_ENABLED"] = "0"
    response = await limiter.mw.dispatch(_request("/admin/x"), _ok)
    assert "X-Admission-State" not in response.headers
    response = await limiter.mw.dispatch(_request("/health"), _ok)
    assert "X-Admission-State" not in response.headers


@pytest.fixture
def signed(monkeypatch):
    """``Bearer tok-a``/``tok-b`` are verified JWTs for tenants escola-a/escola-b."""
    import app.api.auth as auth

    claims = {t: auth.AuthContext(authenticated=True, method="jwt", tenant_id=f"escola-{t[-1]}") for t in ("tok-a", "tok-b")}
    monkeypatch.setattr(auth, "_auth_from_jwt", lambda tok: claims.get(tok))
    monkeypatch.setattr(auth, "_auth_from_api_key", lambda tok: None)
    return lambda t: {"Authorization": f"Bearer {t}"}


@pytest.mark.asyncio
async def test_per_tenant_buckets_are_isolated(limiter, signed):
    mw = limiter.mw
    first = await mw.dispatch(_request("/admin/s", signed("tok-a")), _ok)
    assert first.headers["X-RateLimit-Limit"] == "2"
    assert first.headers["X-RateLimit-Window"] == "15"
    await mw.dispatch(_request("/admin/s", signed("tok-a")), _ok)
    blocked = await mw.dispatch(_request("/admin/s", signed("tok-a")), _ok)
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "2"  # elevated → short retry
    other = await mw.dispatch(_request("/admin/s", signed("tok-b")), _ok)
    assert other.status_code == 200
    assert (await mw.dispatch(_request("/admin/s"), _ok)).status_code == 200  # IP fallback bucket


@pytest.mark.asyncio
async def test_client_chosen_tenant_headers_and_query_share_the_ip_bucket(limiter):
    """Regressão: rodar X-Tenant-ID / ?tenant_id= a cada pedido fugia à quota do limitador adaptativo."""
    mw = limiter.mw
    await mw.dispatch(_request("/admin/s", {"X-Tenant-ID": "escola-a"}), _ok)
    await mw.dispatch(_request("/admin/s", {"X-School-ID": "escola-b"}), _ok)
    blocked = await mw.dispatch(_request("/admin/s", query_string=b"tenant_id=escola-c"), _ok)
    assert blocked.status_code == 429
    assert (await mw.dispatch(_request("/admin/s", {"X-Tenant": "x"}, client="10.0.0.2"), _ok)).status_code == 200


@pytest.mark.asyncio
async def test_congested_rejection_asks_for_longer_retry(limiter):
    limiter.state["snapshot"] = CONGESTED
    limiter.values["ADAPTIVE_LIMITER_ADMIN_PER_SLOT_CONGESTED"] = "1"
    await limiter.mw.dispatch(_request("/ops/x"), _ok)
    blocked = await limiter.mw.dispatch(_request("/ops/x"), _ok)
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "5"


@pytest.mark.asyncio
async def test_interactive_over_quota_is_deferred_to_job_not_rejected(limiter, monkeypatch):
    monkeypatch.setattr(rl.RateLimitMiddleware, "_should_preempt_to_async", lambda *a: False)
    limiter.values["ADAPTIVE_LIMITER_INTERACTIVE_PER_SLOT_ELEVATED"] = "1"
    first = _request("/query", {"X-Tenant": "t1"})
    await limiter.mw.dispatch(first, _ok)
    assert getattr(first.state, "defer_to_query_job", False) is False
    second = _request("/query", {"X-Tenant": "t1"})
    response = await limiter.mw.dispatch(second, _ok)
    assert response.status_code == 200
    assert second.state.defer_to_query_job is True
    assert second.state.query_job_reason == "ollama_overloaded"
    assert response.headers["X-RateLimit-Reason"] == "ollama_overloaded"
