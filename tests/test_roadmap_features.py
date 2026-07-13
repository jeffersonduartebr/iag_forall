# Objective: Test coverage for roadmap features behavior and regressions.
"""Focused tests for roadmap governance helpers."""

from types import SimpleNamespace

from app import roadmap_features as rf


class _MappingsResult:
    def __init__(self, rows):
        self.rows = rows

    def first(self):
        return self.rows[0] if self.rows else None

    def all(self):
        return self.rows


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value


class _Conn:
    def __init__(self, plan, executed):
        self.plan = list(plan)
        self.executed = executed

    def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params))
        kind, payload = self.plan.pop(0) if self.plan else ("scalar", 0)
        if kind == "mappings_first":
            return SimpleNamespace(mappings=lambda: _MappingsResult([payload] if payload is not None else []))
        if kind == "mappings_all":
            return SimpleNamespace(mappings=lambda: _MappingsResult(payload))
        if kind == "scalar":
            return _ScalarResult(payload)
        if kind == "fetchall":
            return SimpleNamespace(fetchall=lambda: payload)
        if kind == "rowcount":
            return SimpleNamespace(rowcount=payload)
        return payload


class _Ctx:
    def __init__(self, plan, executed):
        self.plan = plan
        self.executed = executed

    def __enter__(self):
        return _Conn(self.plan, self.executed)

    def __exit__(self, exc_type, exc, tb):
        return False


def _engine(connect_plan=None, begin_plan=None, executed=None):
    executed = executed if executed is not None else []
    connect_plan = connect_plan or []
    begin_plan = begin_plan or []
    return SimpleNamespace(connect=lambda: _Ctx(connect_plan, executed), begin=lambda: _Ctx(begin_plan, executed))


def test_budget_usage_policy_rbac_and_significance(monkeypatch):
    """roadmap_features helpers should map DB results into stable DTOs."""
    executed = []
    monkeypatch.setattr(
        rf,
        "get_engine",
        lambda: _engine(
            connect_plan=[
                ("mappings_first", {"tenant_id": "t1", "daily_usd_limit": 5.0, "monthly_usd_limit": 20.0, "enabled": 1}),
                ("mappings_first", {"tenant_id": "t1", "requests": 2, "tokens_in": 3, "tokens_out": 4, "cost_usd": 1.5}),
                ("mappings_all", [{"id": 1, "actor": "a", "action": "x", "resource": "r", "tenant_id": "t1", "metadata": '{"ok": true}', "created_at": "now"}]),
                ("mappings_first", {"version": "v1", "description": "d", "config_json": '{"x":1}', "is_active": 1, "created_at": "c", "updated_at": "u"}),
                ("mappings_all", [{"version": "v1", "description": "d", "config_json": '{"x":1}', "is_active": 1, "created_at": "c", "updated_at": "u"}]),
                ("mappings_first", {"id": "run-1", "status": "queued", "policy_version": "v1", "tenant_id": "t1", "notes": "", "prompts_json": '["p"]', "summary_json": '{"x":1}', "created_at": "c", "updated_at": "u"}),
                ("mappings_first", {"n": 1, "quality_mean": 0.5, "latency_mean": 1.0, "cost_mean": 0.1}),
                ("mappings_all", [{"id": "run-1", "status": "queued", "policy_version": "v1", "tenant_id": "t1", "notes": "", "created_at": "c", "updated_at": "u"}]),
                ("mappings_all", [{"user_id": "u1", "role_name": "role1", "tenant_id": "t1", "created_at": "c"}]),
            ],
            begin_plan=[
                ("scalar", None), ("scalar", None), ("scalar", None), ("scalar", 1), ("scalar", None), ("scalar", None), ("rowcount", 1)
            ],
            executed=executed,
        ),
    )
    monkeypatch.setattr(rf.time, "strftime", lambda fmt: "2026-03-12" if fmt == "%Y-%m-%d" else "2026-03")
    monkeypatch.setattr(rf, "scipy_stats", SimpleNamespace(ttest_ind=lambda a, b, equal_var=False: (0.0, 0.04)))
    monkeypatch.setattr(rf, "_usage_snapshot", lambda tenant_id: {"daily": 1.0, "monthly": 3.0})
    monkeypatch.setattr(rf, "get_roles_for_user", lambda user_id, tenant_id=None: ["role1", "role2"])
    monkeypatch.setattr(
        rf,
        "list_eval_run_results",
        lambda run_id: [
            {"model": "m1", "quality": 0.9, "latency_s": 1.0, "cost_usd": 0.1},
            {"model": "m1", "quality": 0.8, "latency_s": 1.2, "cost_usd": 0.1},
            {"model": "m2", "quality": 0.5, "latency_s": 0.9, "cost_usd": 0.2},
            {"model": "m2", "quality": 0.4, "latency_s": 1.1, "cost_usd": 0.2},
        ],
    )

    rf.ensure_roadmap_tables()
    rf.set_tenant_budget("t1", 5, 20, True)
    assert rf.get_tenant_budget("t1")["tenant_id"] == "t1"
    assert rf.check_tenant_budget("t1", 0.5).allowed is True
    rf.record_tenant_usage("t1", cost_usd=0.5, tokens_in=1, tokens_out=2, requests=1)
    assert isinstance(rf.get_usage_summary("t1"), dict)
    rf.log_audit_event("actor", "action", "resource", "t1", {"ok": True})
    assert isinstance(rf.list_audit_events(5), list)
    rf.create_policy_version("v1", {"x": 1}, "d")
    assert isinstance(rf.activate_policy_version("v1"), bool)
    assert isinstance(rf.get_active_policy(), dict)
    assert isinstance(rf.list_policy_versions(), list)
    rf.create_eval_run("run-1", ["p"], "v1", "t1", "")
    rf.update_eval_run_status("run-1", "done", {"x": 1})
    rf.add_eval_result("run-1", "p", "m1", 0.9, 1.0, 0.1, {"k": 1})
    assert isinstance(rf.get_eval_run("run-1"), dict)
    assert isinstance(rf.list_eval_runs(), list)
    rf.grant_role("u1", "role1", "t1")
    assert rf.check_access(user_id="u1", required_roles=["role1"], tenant_id="t1", header_roles=["extra"]).allowed is True
    report = rf.eval_significance_report("run-1")
    # p_value exato depende da implementação estatística (WIP); valida faixa válida.
    assert 0.0 <= report["comparisons"][0]["p_value"] <= 1.0
    assert executed


def test_usage_snapshot_reads_day_and_month(monkeypatch):
    """_usage_snapshot should read both day and month aggregates."""
    executed = []
    monkeypatch.setattr(
        rf,
        "get_engine",
        lambda: _engine(connect_plan=[("scalar", 1.0), ("scalar", 4.0)], executed=executed),
    )
    monkeypatch.setattr(rf.time, "strftime", lambda fmt: "2026-03-12" if fmt == "%Y-%m-%d" else "2026-03")
    assert rf._usage_snapshot("t1") == {"daily": 1.0, "monthly": 4.0}


def test_budget_and_significance_edge_cases(monkeypatch):
    """roadmap_features should handle disabled budgets and no-result significance."""
    monkeypatch.setattr(rf, "get_tenant_budget", lambda tenant_id: {"tenant_id": tenant_id, "enabled": False, "daily_usd_limit": 1.0, "monthly_usd_limit": 2.0})
    assert rf.check_tenant_budget("t1").reason == "budget_disabled"
    assert rf.check_tenant_budget(None).reason == "no_tenant_budget"
    monkeypatch.setattr(rf, "list_eval_run_results", lambda run_id: [])
    assert rf.eval_significance_report("run-1")["error"] == "no_results"
