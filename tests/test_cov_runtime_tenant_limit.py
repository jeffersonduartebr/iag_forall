# Objective: Steady-state per-tenant quotas — isolation, JWT identity precedence, Redis-down policy.
"""Coverage of app.middleware.tenant_rate_limit over the real Redis sliding window (fakeredis)."""

from __future__ import annotations

from types import SimpleNamespace

import app.middleware.tenant_rate_limit as tl
import app.utils.redis_distributed as rd
import pytest
from starlette.requests import Request
from starlette.responses import Response


class _Settings:
    def __init__(self, values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


def _request(path="/query", headers=None, client="10.0.0.9"):
    scope = {"type": "http", "method": "POST", "path": path, "query_string": b"", "scheme": "http",
             "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
             "client": (client, 1), "server": ("t", 80)}
    return Request(scope)


async def _ok(_request):
    return Response("ok")


@pytest.fixture
def tenant(monkeypatch, fake_aioredis, fake_clock):
    values = {"TENANT_RATE_LIMIT_RPM": "2", "TENANT_RATE_LIMIT_WINDOW_S": "60"}
    monkeypatch.setattr(tl, "settings", _Settings(values))
    monkeypatch.setattr(rd, "settings", _Settings(values))
    return SimpleNamespace(values=values, clock=fake_clock(rd), mw=tl.TenantRateLimitMiddleware(app=None))


async def _hit(mw, clock, n, **kw):
    out = []
    for _ in range(n):
        out.append(await mw.dispatch(_request(**kw), _ok))
        clock.advance(0.01)  # distinct sorted-set members
    return out


@pytest.fixture
def jwt_claims(monkeypatch):
    """Signed tokens stand in for real JWTs: ``Bearer tok-a`` carries tenant ``escola-a``."""
    import app.api.auth as auth

    claims = {
        "tok-a": auth.AuthContext(authenticated=True, method="jwt", tenant_id="escola-a", user_id="u1"),
        "tok-b": auth.AuthContext(authenticated=True, method="jwt", tenant_id="escola-b", user_id="u2"),
        "tok-u": auth.AuthContext(authenticated=True, method="jwt", user_id="u9"),
    }
    monkeypatch.setattr(auth, "_auth_from_jwt", lambda tok: claims.get(tok))
    monkeypatch.setattr(auth, "_auth_from_api_key", lambda tok: None)
    return claims


@pytest.mark.asyncio
async def test_tenants_have_independent_quotas_and_window_expires(tenant, jwt_claims):
    a = await _hit(tenant.mw, tenant.clock, 3, headers={"Authorization": "Bearer tok-a"})
    assert [r.status_code for r in a] == [200, 200, 429]
    assert a[0].headers["X-Tenant-RateLimit-Limit"] == "2"
    assert a[2].headers["Retry-After"] == "60"
    b = await _hit(tenant.mw, tenant.clock, 1, headers={"Authorization": "Bearer tok-b"})
    assert b[0].status_code == 200
    tenant.clock.advance(61)
    assert (await _hit(tenant.mw, tenant.clock, 1, headers={"Authorization": "Bearer tok-a"}))[0].status_code == 200


@pytest.mark.asyncio
async def test_rotating_tenant_headers_does_not_escape_the_quota(tenant):
    """Regressão: sem JWT, cada X-Tenant-ID novo abria um balde novo e a quota nunca se aplicava."""
    rotated = []
    for i, h in enumerate(("X-Tenant-ID", "X-Tenant", "X-School-ID")):
        rotated += await _hit(tenant.mw, tenant.clock, 1, headers={h: f"t{i}"})
    assert [r.status_code for r in rotated] == [200, 200, 429]


@pytest.mark.asyncio
async def test_anonymous_callers_are_bucketed_by_ip(tenant):
    same_ip = await _hit(tenant.mw, tenant.clock, 3)
    assert same_ip[-1].status_code == 429
    assert (await _hit(tenant.mw, tenant.clock, 1, client="10.0.0.10"))[0].status_code == 200


@pytest.mark.asyncio
async def test_only_verified_auth_names_the_tenant(tenant, jwt_claims):
    mw = tenant.mw
    assert await mw._resolve_tenant(_request(headers={"Authorization": "Bearer tok-a", "X-Tenant-ID": "fake"})) == "escola-a"
    assert await mw._resolve_tenant(_request(headers={"Authorization": "Bearer tok-u"})) == "user:u9"
    assert await mw._resolve_tenant(_request(headers={"Authorization": "Bearer bad", "X-Tenant": "h"})) is None
    assert await mw._resolve_tenant(_request(headers={"X-School-ID": "h"})) is None


@pytest.mark.asyncio
async def test_disabled_or_non_query_paths_are_not_counted(tenant, fake_aioredis):
    assert "X-Tenant-RateLimit-Limit" not in (await tenant.mw.dispatch(_request("/admin/x"), _ok)).headers
    tenant.values["TENANT_RATE_LIMIT_ENABLED"] = "off"
    assert "X-Tenant-RateLimit-Limit" not in (await tenant.mw.dispatch(_request(), _ok)).headers
    assert await fake_aioredis.keys("tenant-rl:*") == []


@pytest.mark.asyncio
async def test_redis_down_fails_open_in_dev_and_closed_in_production(tenant, monkeypatch):
    async def _down(_build):
        return None

    monkeypatch.setattr("app.utils.redis_async_ops.redis_pipeline_execute", _down)
    assert (await tenant.mw.dispatch(_request(), _ok)).status_code == 200
    tenant.values["ENV"] = "production"
    with pytest.raises(RuntimeError, match="Redis"):
        await tenant.mw.dispatch(_request(), _ok)


def test_bad_settings_fall_back_to_defaults(monkeypatch):
    monkeypatch.setattr(tl, "settings", _Settings({"TENANT_RATE_LIMIT_RPM": "x", "TENANT_RATE_LIMIT_WINDOW_S": "1"}))
    assert tl._rpm() == 120
    assert tl._window_s() == 10  # floor keeps a sane window
    monkeypatch.setattr(tl, "settings", _Settings({"TENANT_RATE_LIMIT_WINDOW_S": "y"}))
    assert tl._window_s() == 60
