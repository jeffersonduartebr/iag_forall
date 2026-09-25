# Objective: require_admin_or_role reports the identity it actually authorized; rate-limit identity is verified.
"""Who a request acts as must be the identity the authorization checked, never a header it ignored."""

from types import SimpleNamespace

import pytest
from app.api import deps
from starlette.requests import Request


def _allow(reason):
    return lambda **kw: SimpleNamespace(allowed=True, roles=["expert_reviewer"], reason=reason)


def _call(**kw):
    base = {"admin_token": None, "user_id": None, "user_roles_header": None, "required_roles": ["expert_reviewer"]}
    return deps.require_admin_or_role(**{**base, **kw})


@pytest.fixture
def no_admin(monkeypatch):
    def _deny(*a, **k):
        raise deps.HTTPException(status_code=401, detail="no")

    monkeypatch.setattr(deps, "require_admin", _deny)
    monkeypatch.setattr(deps, "_roles_from_jwt", lambda a: (None, None, []))


def test_rbac_result_names_the_user_rbac_checked(no_admin, monkeypatch):
    monkeypatch.setattr(deps, "check_access", _allow("rbac"))
    assert _call(user_id="ana")["user_id"] == "ana"
    assert _call(user_id="bia")["user_id"] == "bia"


def test_header_is_not_the_identity_when_jwt_roles_authorized(no_admin, monkeypatch):
    monkeypatch.setattr(deps, "check_access", _allow("jwt"))
    assert _call(user_id="vitima")["user_id"] is None


def test_jwt_subject_wins(monkeypatch, no_admin):
    monkeypatch.setattr(deps, "_roles_from_jwt", lambda a: ("sub-1", None, ["expert_reviewer"]))
    monkeypatch.setattr(deps, "check_access", _allow("jwt"))
    assert _call(user_id="outro", authorization="Bearer x")["user_id"] == "sub-1"


def test_admin_token_has_its_own_identity(monkeypatch):
    monkeypatch.setattr(deps, "require_admin", lambda t, a: {"username": "admin"})
    out = _call(admin_token="t", user_id="vitima")
    assert (out["authorized_by"], out["user_id"]) == ("admin_token", "admin")


def _request(headers):
    scope = {"type": "http", "method": "GET", "path": "/query", "query_string": b"", "scheme": "http",
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
             "client": ("10.1.1.1", 1), "server": ("t", 80)}
    return Request(scope)


def test_verified_identity_uses_api_key_and_ignores_tenant_headers(monkeypatch):
    import app.api.auth as auth
    from app.middleware import trusted_identity as ti

    key = auth.AuthContext(authenticated=True, method="api_key", api_key_hint="abc")
    monkeypatch.setattr(auth, "_auth_from_jwt", lambda tok: None)
    monkeypatch.setattr(auth, "_auth_from_api_key", lambda tok: key if tok == "k1" else None)
    assert ti.verified_identity(_request({"X-Api-Key": "k1", "X-Tenant-ID": "fake"})) == "key:abc"
    assert ti.verified_identity(_request({"X-Tenant-ID": "fake"})) is None
    assert ti.client_ip(_request({"X-Forwarded-For": "9.9.9.9"})) == "10.1.1.1"


def test_verified_identity_survives_auth_errors_and_anonymous_contexts(monkeypatch):
    import app.api.auth as auth
    from app.middleware import trusted_identity as ti

    def boom(**kw):
        raise RuntimeError("settings down")

    monkeypatch.setattr(ti, "resolve_auth", boom)
    assert ti.verified_identity(_request({"Authorization": "Bearer x"})) is None
    monkeypatch.setattr(ti, "resolve_auth", lambda **kw: auth.AuthContext(authenticated=True, method="admin_token"))
    assert ti.verified_identity(_request({"Authorization": "Bearer x"})) is None
