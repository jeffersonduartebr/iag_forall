# Objective: HTTP-level coverage for the expert portal routes, including role enforcement.
"""expert_routes: which roles reach which route, identity resolution, and error mapping."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

RUBRIC = {k: 7.0 for k in ("factual_correctness", "task_completion", "clarity_structure")}
RUBRIC.update({k: 6.0 for k in ("scaffolding", "audience_fit", "misconception_handling")})
ASSESSMENT = {
    "benchmark_id": "b1",
    "theme": "hist",
    "query_text": "Q",
    "answer": "A",
    "quality_score": 8,
    "rubric": RUBRIC,
}


def _fake_authorizer(*, admin_token, user_id, user_roles_header, authorization, required_roles):
    """Same contract as deps.require_admin_or_role: roles come from X-User-Roles."""
    roles = [r.strip() for r in (user_roles_header or "").split(",") if r.strip()]
    if not any(r in roles for r in required_roles):
        raise HTTPException(status_code=403, detail={"required_roles": required_roles})
    return {"authorized_by": "rbac", "user_id": user_id, "roles": roles}


@pytest.fixture
def api(monkeypatch):
    from app.api import expert_routes as routes

    monkeypatch.setattr("app.api.dependencies.require_admin_or_role", _fake_authorizer)
    monkeypatch.setattr(routes, "_roles_from_jwt", lambda a: ((a or "").removeprefix("Bearer ") or None, None, []))
    audit = []
    monkeypatch.setattr(routes, "log_audit_event", lambda **kw: audit.append(kw))
    monkeypatch.setattr(routes, "ensure_expert_profile", lambda uid: {"user_id": uid, "theme_ids": ["hist"]})
    # Serviços de leitura sem BD por omissão; cada teste substitui o que verifica.
    monkeypatch.setattr(routes, "list_expert_accounts_public", lambda: [])
    monkeypatch.setattr(routes, "list_available_themes", lambda: [])
    monkeypatch.setattr(routes, "expert_judge_agreement_report", lambda **kw: {"kappa": None})
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app), routes, audit


def _h(roles, user=None, jwt=None):
    headers = {"X-User-Roles": roles}
    if user:
        headers["X-User-Id"] = user
    if jwt:
        headers["Authorization"] = f"Bearer {jwt}"
    return headers


@pytest.mark.parametrize(
    ("method", "path", "allowed", "denied"),
    [
        ("get", "/admin/experts/accounts", "governance_admin", "expert_reviewer"),
        ("post", "/admin/experts/accounts", "admin", "researcher"),
        ("put", "/admin/experts/accounts/1", "eval_admin", "audit_viewer"),
        ("get", "/admin/experts/themes", "audit_viewer", "governance_admin"),
        ("get", "/admin/experts/metrics/kappa", "eval_viewer", "expert_reviewer"),
        ("post", "/admin/experts/assessments", "expert_reviewer", "eval_viewer"),
    ],
)
def test_role_enforcement_happens_before_body_validation(api, method, path, allowed, denied):
    client, _, _ = api
    denied_resp = getattr(client, method)(path, headers=_h(denied), **({"json": {}} if method != "get" else {}))
    assert denied_resp.status_code == 403  # nunca 422: o anónimo não aprende o esquema
    if method == "get":
        assert getattr(client, method)(path, headers=_h(allowed)).status_code == 200


def test_list_and_create_accounts(api, monkeypatch):
    client, routes, audit = api
    monkeypatch.setattr(routes, "list_expert_accounts_public", lambda: [{"id": 1}])
    assert client.get("/admin/experts/accounts", headers=_h("admin")).json() == {"items": [{"id": 1}]}

    monkeypatch.setattr(routes, "register_expert_account", lambda **kw: {"id": 2, "email": kw["email"]})
    body = {"display_name": "Ana", "email": "a@x.org", "password": "senha1234"}
    resp = client.post("/admin/experts/accounts", json=body, headers=_h("admin", user="gestor"))
    assert resp.json() == {"status": "created", "account": {"id": 2, "email": "a@x.org"}}
    assert audit[-1]["actor"] == "gestor" and audit[-1]["action"] == "expert_account_create"
    client.post("/admin/experts/accounts", json=body, headers=_h("admin"))
    assert audit[-1]["actor"] == "rbac" and audit[-1]["metadata"]["roles"] == ["admin"]


def test_audit_actor_is_the_authenticated_identity_not_the_header(api, monkeypatch):
    """Um admin autenticado pelo token não pode assinar a auditoria como outra pessoa via X-User-Id."""
    client, routes, audit = api
    monkeypatch.setattr(
        "app.api.dependencies.require_admin_or_role",
        lambda **kw: {"authorized_by": "admin_token", "user_id": "admin", "roles": ["admin"]},
    )
    monkeypatch.setattr(routes, "register_expert_account", lambda **kw: {"id": 2, "email": kw["email"]})
    monkeypatch.setattr(routes, "update_expert_account_admin", lambda account_id, **kw: {"id": 3, "email": "c@x"})
    body = {"display_name": "Ana", "email": "a@x.org", "password": "senha1234"}
    client.post("/admin/experts/accounts", json=body, headers={"X-User-Id": "vitima"})
    assert audit[-1]["actor"] == "admin"
    client.put("/admin/experts/accounts/3", json={"enabled": False}, headers={"X-User-Id": "vitima"})
    assert audit[-1]["actor"] == "admin"


def test_create_account_value_error_is_400(api, monkeypatch):
    client, routes, audit = api

    def boom(**kw):
        raise ValueError("E-mail já cadastrado.")

    monkeypatch.setattr(routes, "register_expert_account", boom)
    body = {"display_name": "Ana", "email": "a@x.org", "password": "senha1234"}
    resp = client.post("/admin/experts/accounts", json=body, headers=_h("admin"))
    assert resp.status_code == 400 and resp.json()["detail"] == "E-mail já cadastrado."
    assert audit == []


def test_update_account_paths(api, monkeypatch):
    client, routes, audit = api
    results = iter([None, {"id": 3, "email": "c@x.org"}])
    monkeypatch.setattr(routes, "update_expert_account_admin", lambda account_id, **kw: next(results))
    assert client.put("/admin/experts/accounts/3", json={"enabled": False}, headers=_h("admin")).status_code == 404
    resp = client.put("/admin/experts/accounts/3", json={"enabled": False}, headers=_h("admin", user="g"))
    assert resp.json()["status"] == "updated"
    assert audit[-1]["metadata"]["account_id"] == 3 and audit[-1]["actor"] == "g"

    def boom(account_id, **kw):
        raise ValueError("Senha deve ter pelo menos 8 caracteres.")

    monkeypatch.setattr(routes, "update_expert_account_admin", boom)
    assert client.put("/admin/experts/accounts/3", json={}, headers=_h("admin")).status_code == 400


def test_profile_uses_signed_identity_not_header(api, monkeypatch):
    client, routes, _ = api
    resp = client.get("/admin/experts/profile", headers=_h("expert_reviewer", user="outro", jwt="eu@x.org"))
    assert resp.json()["user_id"] == "eu@x.org"
    # Sem JWT, a identidade é o X-User-Id que a RBAC verificou.
    assert client.get("/admin/experts/profile", headers=_h("expert_reviewer", user="svc")).json()["user_id"] == "svc"


def test_rbac_experts_without_jwt_do_not_share_one_identity(api):
    """Regressão: todo perito autorizado por RBAC virava o literal "rbac" (perfil e avaliações partilhados)."""
    client, _, _ = api
    ana = client.get("/admin/experts/profile", headers=_h("expert_reviewer", user="ana")).json()["user_id"]
    bia = client.get("/admin/experts/profile", headers=_h("expert_reviewer", user="bia")).json()["user_id"]
    assert (ana, bia) == ("ana", "bia")


def test_no_authorized_identity_is_refused_not_pooled(api):
    client, _, _ = api
    resp = client.get("/admin/experts/profile", headers=_h("expert_reviewer"))
    assert resp.status_code == 403


def test_put_profile_updates_and_audits(api, monkeypatch):
    client, routes, audit = api
    seen = {}
    monkeypatch.setattr(routes, "update_expert_profile", lambda uid, **kw: seen.update(uid=uid, **kw) or {"ok": 1})
    body = {"display_name": "Ana", "theme_ids": ["hist", "fis"]}
    assert client.put("/admin/experts/profile", json=body, headers=_h("researcher", jwt="ana")).json() == {"ok": 1}
    assert seen == {"uid": "ana", "display_name": "Ana", "theme_ids": ["hist", "fis"], "credentials_note": None}
    assert audit[-1] == {
        "actor": "ana",
        "action": "expert_profile_update",
        "resource": "expert_profiles",
        "metadata": {"theme_ids": ["hist", "fis"], "roles": ["researcher"]},
    }


def test_themes_and_next_item(api, monkeypatch):
    client, routes, _ = api
    monkeypatch.setattr(routes, "list_available_themes", lambda: [{"id": "hist"}])
    assert client.get("/admin/experts/themes", headers=_h("eval_admin")).json() == {"items": [{"id": "hist"}]}

    calls = []
    items = iter([None, {"benchmark_id": "b1"}])
    monkeypatch.setattr(routes, "get_next_review_item", lambda uid, **kw: calls.append((uid, kw)) or next(items))
    headers = _h("expert_reviewer", jwt="ana")
    assert client.get("/admin/experts/next-item", headers=headers).json() == {"status": "empty", "item": None}
    ok = client.get("/admin/experts/next-item?eval_run_id=r1&split=all&seed=4", headers=headers).json()
    assert ok == {"status": "ok", "item": {"benchmark_id": "b1"}}
    assert calls[1] == ("ana", {"eval_run_id": "r1", "split": "all", "seed": 4})


def test_assessment_theme_outside_profile_is_forbidden(api, monkeypatch):
    client, routes, audit = api
    monkeypatch.setattr(routes, "submit_expert_assessment", lambda *a, **k: pytest.fail("não devia gravar"))
    headers = _h("expert_reviewer", user="ana")
    resp = client.post("/admin/experts/assessments", json={**ASSESSMENT, "theme": "fis"}, headers=headers)
    assert resp.status_code == 403 and "fis" in resp.json()["detail"]
    assert audit == []


def test_assessment_saved_and_listed(api, monkeypatch):
    client, routes, audit = api
    saved = {}
    monkeypatch.setattr(routes, "submit_expert_assessment", lambda uid, **kw: saved.update(uid=uid, **kw) or {"id": 5})
    resp = client.post("/admin/experts/assessments", json=ASSESSMENT, headers=_h("expert_reviewer", jwt="ana"))
    assert resp.json() == {"status": "saved", "assessment": {"id": 5}}
    assert saved["uid"] == "ana" and saved["rubric"]["scaffolding"] == 6.0 and saved["quality_score"] == 8.0
    assert audit[-1]["action"] == "expert_assessment_submit" and audit[-1]["metadata"]["benchmark_id"] == "b1"

    # Perfil sem temas: qualquer tema é aceite.
    monkeypatch.setattr(routes, "ensure_expert_profile", lambda uid: {"theme_ids": []})
    assert (
        client.post(
            "/admin/experts/assessments", json={**ASSESSMENT, "theme": "fis"}, headers=_h("expert_reviewer", user="a")
        ).status_code
        == 200
    )

    monkeypatch.setattr(routes, "list_expert_assessments", lambda **kw: [kw])
    listed = client.get("/admin/experts/assessments?theme=hist&limit=3", headers=_h("audit_viewer", jwt="ana")).json()
    assert listed == {"items": [{"expert_id": "ana", "theme": "hist", "eval_run_id": None, "limit": 3}]}


def test_kappa_metrics_forwards_filters(api, monkeypatch):
    client, routes, _ = api
    monkeypatch.setattr(routes, "expert_judge_agreement_report", lambda **kw: {"kappa": 0.5, **kw})
    resp = client.get("/admin/experts/metrics/kappa?eval_run_id=r&theme=t", headers=_h("eval_viewer"))
    assert resp.json() == {"kappa": 0.5, "eval_run_id": "r", "theme": "t"}


@pytest.mark.parametrize(("extra", "hints"), [({}, None), ({"theme": "hist"}, {"theme": "hist", "benchmark_id": None})])
def test_preview_answer_runs_router_without_cache(api, monkeypatch, extra, hints):
    client, _, _ = api
    seen = []

    async def fake_process(req):
        seen.append(req)
        return {"result": {"answer": "resp", "model": "m1", "latency_s": 0.2, "metadata": {"quality": "7.5"}}}

    monkeypatch.setattr("app.services.query_runtime.process_query_request", fake_process)
    resp = client.post(
        "/admin/experts/preview-answer", json={"query": "Q?", **extra}, headers=_h("expert_reviewer", jwt="ana")
    )
    body = resp.json()
    assert (body["answer"], body["model"], body["judge_quality"], body["reviewer"]) == ("resp", "m1", 7.5, "ana")
    assert seen[0].use_cache is False and seen[0].query == "Q?"
    got = seen[0].workload_hints
    assert (got is None and hints is None) or {"theme": got.theme, "benchmark_id": got.benchmark_id} == hints


def test_preview_answer_honours_frozen_policy(api, monkeypatch, fake_redis):
    from app.services import frozen_policy as fp

    client, _, _ = api
    monkeypatch.setattr(fp, "build_frozen_snapshot", lambda: {"BANDIT_EPSILON": 0.1})
    active = []

    async def fake_process(req):
        active.append(fp.is_frozen_policy_active())
        return {"result": {"answer": "a", "metadata": {}}}

    monkeypatch.setattr("app.services.query_runtime.process_query_request", fake_process)
    for flag in (True, False):
        body = client.post(
            "/admin/experts/preview-answer",
            json={"query": "Q?", "frozen_policy": flag},
            headers=_h("expert_reviewer", jwt="ana"),
        ).json()
        assert body["frozen_policy"]["active"] is flag
    assert active == [True, False] and fp.is_frozen_policy_active() is False
