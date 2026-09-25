# Objective: Budget, audit, policy and RBAC helpers of roadmap_features against real SQL (SQLite).
"""roadmap_features governance: tenant budgets, audit log, policy versions and role checks."""

import time

import pytest
from sqlalchemy import create_engine, text  # vinculado na coleta, antes do mock global de create_engine
from sqlalchemy.pool import StaticPool

_DDL = (
    "CREATE TABLE tenant_budgets (tenant_id TEXT PRIMARY KEY, daily_usd_limit REAL, monthly_usd_limit REAL,"
    " enabled INTEGER, updated_at TEXT)",
    "CREATE TABLE tenant_usage (id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT, day_key TEXT, month_key TEXT,"
    " requests INTEGER DEFAULT 0, tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0, cost_usd REAL DEFAULT 0)",
    "CREATE TABLE audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT, action TEXT, resource TEXT,"
    " tenant_id TEXT, metadata TEXT, created_at TEXT)",
    "CREATE TABLE policy_versions (id INTEGER PRIMARY KEY AUTOINCREMENT, version TEXT UNIQUE, description TEXT,"
    " config_json TEXT, is_active INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT)",
    "CREATE TABLE rbac_user_roles (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, role_name TEXT,"
    " tenant_id TEXT, created_at TEXT)",
)


@pytest.fixture
def rf(monkeypatch):
    from app import roadmap_features

    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with eng.begin() as conn:
        for ddl in _DDL:
            conn.execute(text(ddl))
    monkeypatch.setattr(roadmap_features, "get_engine", lambda: eng)
    for cache in ("_HOTPATH_POLICY_CACHE", "_HOTPATH_TENANT_BUDGET_CACHE", "_HOTPATH_TENANT_USAGE_CACHE"):
        getattr(roadmap_features, cache).clear()
    monkeypatch.setattr(roadmap_features, "engine", eng, raising=False)
    return roadmap_features


def _sql(rf, sql, **params):
    with rf.engine.begin() as conn:
        conn.execute(text(sql), params)


def _budget(rf, tenant, daily, monthly, enabled=1):
    _sql(rf, "INSERT INTO tenant_budgets VALUES (:t, :d, :m, :e, NULL)", t=tenant, d=daily, m=monthly, e=enabled)


def _usage(rf, tenant, cost, day=None, month=None):
    _sql(
        rf,
        "INSERT INTO tenant_usage (tenant_id, day_key, month_key, requests, cost_usd) VALUES (:t, :d, :m, 1, :c)",
        t=tenant,
        d=day or time.strftime("%Y-%m-%d"),
        m=month or time.strftime("%Y-%m"),
        c=cost,
    )


def test_budget_lookup_defaults_and_cache(rf):
    assert rf.get_tenant_budget("ghost") == {
        "tenant_id": "ghost",
        "daily_usd_limit": 0.0,
        "monthly_usd_limit": 0.0,
        "enabled": False,
    }
    _budget(rf, "ghost", 1.0, 2.0)
    assert rf.get_tenant_budget("ghost")["enabled"] is False  # ainda em cache
    rf._HOTPATH_TENANT_BUDGET_CACHE.clear()
    assert rf.get_tenant_budget("ghost")["daily_usd_limit"] == 1.0
    assert [b["tenant_id"] for b in rf.list_tenant_budgets()] == ["ghost"]


def test_list_budgets_survives_missing_table(rf):
    _sql(rf, "DROP TABLE tenant_budgets")
    assert rf.list_tenant_budgets() == []


@pytest.mark.parametrize(
    ("daily", "monthly", "today", "earlier", "projected", "reason", "allowed"),
    [
        (5.0, 50.0, 1.0, 0.0, 0.5, "ok", True),
        (1.0, 50.0, 0.8, 0.0, 0.5, "daily_limit_exceeded", False),
        (0.0, 3.0, 1.0, 2.5, 0.0, "monthly_limit_exceeded", False),
        (0.0, 0.0, 99.0, 99.0, 9.0, "ok", True),  # limite 0 = sem limite
    ],
)
def test_check_tenant_budget(rf, daily, monthly, today, earlier, projected, reason, allowed):
    _budget(rf, "t1", daily, monthly)
    _usage(rf, "t1", today)
    _usage(rf, "t1", earlier, day="1999-01-01")  # outro dia, mesmo mês
    check = rf.check_tenant_budget("t1", projected_cost_usd=projected)
    assert (check.reason, check.allowed) == (reason, allowed)
    assert check.daily_spent == pytest.approx(today)
    assert check.monthly_spent == pytest.approx(today + earlier)


def test_check_budget_without_tenant_or_disabled(rf):
    assert rf.check_tenant_budget(None).reason == "no_tenant_budget"
    _budget(rf, "off", 1.0, 2.0, enabled=0)
    check = rf.check_tenant_budget("off", 100.0)
    assert (check.allowed, check.reason, check.daily_limit, check.monthly_limit) == (True, "budget_disabled", 1.0, 2.0)


def test_usage_snapshot_is_cached(rf):
    _usage(rf, "t1", 1.0)
    assert rf._usage_snapshot("t1") == {"daily": 1.0, "monthly": 1.0}
    _usage(rf, "t1", 1.0, day="1999-01-01")
    assert rf._usage_snapshot("t1") == {"daily": 1.0, "monthly": 1.0}


def test_usage_summary_per_tenant_and_global(rf):
    _usage(rf, "a", 1.5)
    _usage(rf, "b", 3.0)
    _usage(rf, "b", 9.0, month="1999-01")  # fora do mês corrente
    assert rf.get_usage_summary("b")["cost_usd"] == pytest.approx(3.0)
    assert rf.get_usage_summary("nobody") == {
        "tenant_id": "nobody",
        "requests": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
    }
    summary = rf.get_usage_summary()
    assert summary["month_key"] == time.strftime("%Y-%m")
    assert [(i["tenant_id"], i["requests"]) for i in summary["items"]] == [("b", 1), ("a", 1)]


def test_audit_log_truncates_and_parses_metadata(rf):
    rf.log_audit_event("x" * 200, "act", "res", tenant_id="t", metadata={"k": "ç"})
    _sql(rf, "INSERT INTO audit_log (actor, action, resource, metadata) VALUES ('a', 'b', 'c', '{ruim')")
    events = rf.list_audit_events(limit=0)
    assert len(events) == 1 and events[0]["metadata"] == {}  # limite mínimo 1, mais recente primeiro
    older = rf.list_audit_events()[1]
    assert len(older["actor"]) == 128 and older["metadata"] == {"k": "ç"} and older["tenant_id"] == "t"


def test_policy_activation_and_listing(rf):
    assert rf.activate_policy_version("v9") is False
    assert rf.get_active_policy() is None
    _sql(rf, "INSERT INTO policy_versions (version, config_json, is_active) VALUES ('v1', '{\"w\": 1}', 1)")
    _sql(rf, "INSERT INTO policy_versions (version, config_json) VALUES ('v2', 'nao-json')")
    assert rf.get_active_policy() is None  # o None ficou em cache
    assert rf.activate_policy_version("v2") is True  # limpa a cache
    active = rf.get_active_policy()
    assert active["version"] == "v2" and active["config"] == {}
    assert rf.get_active_policy() is active  # servido da cache
    listed = rf.list_policy_versions()
    assert [(p["version"], p["is_active"], p["config"]) for p in listed] == [("v2", 1, {}), ("v1", 0, {"w": 1})]


def test_active_policy_config_is_parsed(rf):
    _sql(rf, "INSERT INTO policy_versions (version, config_json, is_active) VALUES ('v1', '{\"w\": 2}', 1)")
    assert rf.get_active_policy()["config"] == {"w": 2}


def _role(rf, user, role, tenant=None):
    _sql(
        rf, "INSERT INTO rbac_user_roles (user_id, role_name, tenant_id) VALUES (:u, :r, :t)", u=user, r=role, t=tenant
    )


def test_roles_listing_resolution_and_revocation(rf):
    _role(rf, "ana", "eval_admin")
    _role(rf, "ana", "researcher", tenant="t1")
    _role(rf, "bia", "expert_reviewer")
    assert sorted(rf.get_roles_for_user("ana")) == ["eval_admin"]
    assert sorted(rf.get_roles_for_user("ana", tenant_id="t1")) == ["eval_admin", "researcher"]
    assert [r["role_name"] for r in rf.list_roles("ana")] == ["researcher", "eval_admin"]
    assert len(rf.list_roles()) == 3
    assert rf.revoke_role("ana", "researcher") == 0  # só existe no tenant t1
    assert rf.revoke_role("ana", "researcher", tenant_id="t1") == 1
    assert rf.revoke_role("ana", "eval_admin") == 1
    assert rf.list_roles("ana") == []


def test_check_access_sources(rf, monkeypatch):
    from app.settings_dynamic import settings

    _role(rf, "ana", "eval_admin")
    assert rf.check_access(user_id=None, required_roles=[]).reason == "no_required_role"
    by_db = rf.check_access(user_id="ana", required_roles=["eval_admin"])
    assert (by_db.allowed, by_db.reason, by_db.roles) == (True, "rbac", ["eval_admin"])
    by_jwt = rf.check_access(user_id=None, required_roles=["researcher"], jwt_roles=[" researcher ", ""])
    assert (by_jwt.allowed, by_jwt.reason, by_jwt.roles) == (True, "jwt", ["researcher"])

    monkeypatch.setattr(settings, "_get_bool", lambda key, default=False: False)
    denied = rf.check_access(user_id="bia", required_roles=["platform_admin"], header_roles=["platform_admin"])
    assert (denied.allowed, denied.reason) == (False, "missing_required_role")

    monkeypatch.setattr(settings, "_get_bool", lambda key, default=False: True)
    trusted = rf.check_access(user_id="bia", required_roles=["platform_admin"], header_roles=["platform_admin", " "])
    assert trusted.allowed and trusted.roles == ["platform_admin"]

    def boom(*a, **k):
        raise RuntimeError("settings em baixo")

    monkeypatch.setattr(settings, "_get_bool", boom)
    assert rf.check_access(user_id="bia", required_roles=["x"], header_roles=["x"]).allowed is False
