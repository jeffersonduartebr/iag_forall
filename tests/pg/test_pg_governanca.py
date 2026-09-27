# Objective: Governance on PostgreSQL: budgets, usage, audit, policies, evals, reviews, RBAC (roadmap_features).
"""Each test calls the real function and reads the rows back (ids in tests/pg/CATALOGO.md)."""

from __future__ import annotations

import app.roadmap_features as rf


def test_tenant_budget_upsert_read_and_list(sql):
    rf.set_tenant_budget("t1", 5.0, 50.0)
    rf.set_tenant_budget("t1", 7.0, 70.0, enabled=False)
    rf.set_tenant_budget("t0", 1.0, 2.0)
    assert rf.get_tenant_budget("t1") == {"tenant_id": "t1", "daily_usd_limit": 7.0, "monthly_usd_limit": 70.0,
                                          "enabled": 0}
    assert rf.get_tenant_budget("nenhum")["enabled"] is False
    assert [b["tenant_id"] for b in rf.list_tenant_budgets()] == ["t0", "t1"]


def test_tenant_usage_sums_on_conflict_and_feeds_the_budget_check(sql):
    rf.set_tenant_budget("t", 1.0, 100.0)
    rf.record_tenant_usage("t", cost_usd=0.4, tokens_in=10, tokens_out=5)
    rf.record_tenant_usage("t", cost_usd=0.5, tokens_in=1, tokens_out=2, requests=2)
    [linha] = sql("SELECT requests, tokens_in, tokens_out, cost_usd, month_key FROM tenant_usage")
    assert (linha["requests"], linha["tokens_in"], linha["tokens_out"]) == (3, 11, 7)
    assert abs(linha["cost_usd"] - 0.9) < 1e-9 and len(linha["month_key"]) == 7
    assert rf._usage_snapshot("t")["daily"] == linha["cost_usd"]
    assert rf.check_tenant_budget("t", projected_cost_usd=0.2).reason == "daily_limit_exceeded"


def test_usage_summary_for_one_tenant_and_for_all(sql):
    rf.record_tenant_usage("a", cost_usd=1.0, tokens_in=3)
    rf.record_tenant_usage("b", cost_usd=2.0, tokens_in=4)
    um = rf.get_usage_summary("a")
    assert um["tenant_id"] == "a" and int(um["tokens_in"]) == 3 and float(um["cost_usd"]) == 1.0
    todos = rf.get_usage_summary(None)
    assert [i["tenant_id"] for i in todos["items"]] == ["b", "a"]
    assert rf.get_usage_summary("vazio")["requests"] == 0


def test_audit_event_round_trip(sql):
    rf.log_audit_event("admin", "settings.update", "X", tenant_id="t", metadata={"de": 1, "para": "ação"})
    [evento] = rf.list_audit_events(10)
    assert evento["actor"] == "admin" and evento["metadata"] == {"de": 1, "para": "ação"}


def test_policy_versions_upsert_activate_and_deactivate(sql):
    rf.create_policy_version("v1", {"a": 1}, "primeira")
    rf.create_policy_version("v2", {"b": 2})
    rf.create_policy_version("v1", {"a": 3}, "revista")
    assert rf.activate_policy_version("v1")
    assert rf.get_active_policy()["config"] == {"a": 3}
    assert rf.activate_policy_version("v2")
    assert rf.get_active_policy()["version"] == "v2"
    assert sql("SELECT version, is_active FROM policy_versions ORDER BY version") == [
        {"version": "v1", "is_active": 0}, {"version": "v2", "is_active": 1}]
    assert rf.activate_policy_version("inexistente") is False
    assert [p["version"] for p in rf.list_policy_versions()] == ["v2", "v1"]
    assert rf.list_policy_versions()[1]["description"] == "revista"


def test_eval_run_header_results_status_and_aggregate(sql):
    rf.create_eval_run("r1", ["p1", "p2"], policy_version="v1", tenant_id="t", metadata={"k": 1})
    rf.add_eval_result("r1", "p1", "m", 8.0, 1.0, 0.01, {"x": 1})
    rf.add_eval_result("r1", "p2", "m", 6.0, 3.0, 0.03)
    sql("UPDATE eval_runs SET updated_at = now() - interval '1 day'")
    rf.update_eval_run_status("r1", "done", {"ok": True})
    run = rf.get_eval_run("r1")
    assert run["status"] == "done" and run["summary"] == {"ok": True} and run["prompts"] == ["p1", "p2"]
    assert run["aggregate"]["n"] == 2 and run["aggregate"]["quality_mean"] == 7.0
    assert rf.get_eval_run("nada") is None
    assert [r["prompt_text"] for r in rf.list_eval_run_results("r1")] == ["p1", "p2"]
    assert rf.list_eval_runs()[0]["metadata"] == {"k": 1}
    [linha] = sql("SELECT updated_at > now() - interval '1 minute' AS tocado FROM eval_runs")
    assert linha["tocado"] is True  # trigger set_col_now('updated_at')


def test_response_review_returns_its_id_and_is_updated(sql):
    kw = dict(correlation_id="c", tenant_id="t", query_text="q", answer="a", chosen_model="m", confidence_score=0.2,
              confidence_band="low", grounded=True, verification_status="unverified", review_reason="low_conf")
    primeiro = rf.create_response_review(**kw)
    segundo = rf.create_response_review(**kw, metadata={"n": 2})
    assert (primeiro, segundo) == (1, 2)
    assert rf.update_response_review(segundo, review_status="approved", reviewer_id="r")
    assert rf.update_response_review(999, review_status="approved") is False
    assert [r["id"] for r in rf.list_response_reviews("approved")] == [2]
    assert [r["id"] for r in rf.list_response_reviews()] == [2, 1]


def test_rbac_grant_is_idempotent_and_revoke_matches_null_tenant(sql):
    rf.grant_role("u", "admin", "t1")
    rf.grant_role("u", "admin", "t1")
    rf.grant_role("u", "reviewer")
    assert sql("SELECT count(*) AS n FROM rbac_user_roles WHERE tenant_id = 't1'") == [{"n": 1}]
    assert sorted(rf.get_roles_for_user("u", "t1")) == ["admin", "reviewer"]
    assert rf.get_roles_for_user("u", "t2") == ["reviewer"]
    assert len(rf.list_roles()) == 2 and len(rf.list_roles("u")) == 2
    assert rf.revoke_role("u", "reviewer", None) == 1  # tenant NULL casa com NULL (IS NOT DISTINCT FROM)
    assert rf.revoke_role("u", "admin", None) == 0  # e NULL não casa com 't1'
    assert rf.revoke_role("u", "admin", "t1") == 1
