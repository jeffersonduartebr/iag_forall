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


@pytest.mark.asyncio
async def test_tenants_have_independent_quotas_and_window_expires(tenant):
    a = await _hit(tenant.mw, tenant.clock, 3, headers={"X-Tenant-ID": "escola-a"})
    assert [r.status_code for r in a] == [200, 200, 429]
    assert a[0].headers["X-Tenant-RateLimit-Limit"] == "2"
    assert a[2].headers["Retry-After"] == "60"
    b = await _hit(tenant.mw, tenant.clock, 1, headers={"X-School-ID": "escola-b"})
    assert b[0].status_code == 200
    tenant.clock.advance(61)
    assert (await _hit(tenant.mw, tenant.clock, 1, headers={"X-Tenant-ID": "escola-a"}))[0].status_code == 200


@pytest.mark.asyncio
async def test_anonymous_callers_are_bucketed_by_ip(tenant):
    same_ip = await _hit(tenant.mw, tenant.clock, 3)
    assert same_ip[-1].status_code == 429
    assert (await _hit(tenant.mw, tenant.clock, 1, client="10.0.0.10"))[0].status_code == 200


@pytest.mark.asyncio
async def test_jwt_identity_wins_over_spoofable_header(tenant, monkeypatch):
    claims = {"tok-t": SimpleNamespace(tenant_id="real", user_id="u1"), "tok-u": SimpleNamespace(tenant_id=None, user_id="u9")}
    monkeypatch.setattr(tl, "_extract_bearer_token", lambda h: (h or "").removeprefix("Bearer ") or None)
    monkeypatch.setattr(tl, "_auth_from_jwt", lambda tok: claims.get(tok))
    mw = tenant.mw
    assert await mw._resolve_tenant(_request(headers={"Authorization": "Bearer tok-t", "X-Tenant-ID": "fake"})) == "real"
    assert await mw._resolve_tenant(_request(headers={"Authorization": "Bearer tok-u"})) == "user:u9"
    assert await mw._resolve_tenant(_request(headers={"Authorization": "Bearer bad", "X-Tenant": "h"})) == "h"


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

    monkeypatch.setattr(rd, "redis_pipeline_execute", _down)
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
