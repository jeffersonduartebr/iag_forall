# Objective: Behavioural coverage for the expert review service (profiles, queues, kappa, dashboard).
"""expert_review: profile merge, catalog/eval queues, submission and agreement reports."""

import pytest


@pytest.fixture
def er():
    # Importado no teste: outros testes fazem sys.modules.pop e trocam o objeto de módulo.
    from app.services import expert_review

    return expert_review


def test_ensure_profile_merges_account_fields(er, monkeypatch):
    monkeypatch.setattr(er, "get_expert_profile", lambda uid: {"user_id": uid, "theme_ids": ["t1"], "display_name": ""})
    account = {"email": "a@x.org", "phone": "9", "enabled": 1, "display_name": "Ana"}
    monkeypatch.setattr(er, "get_expert_account_by_email", lambda uid: account)
    out = er.ensure_expert_profile("a@x.org")
    assert out == {
        "user_id": "a@x.org",
        "theme_ids": ["t1"],
        "display_name": "Ana",  # vazio no perfil -> vem da conta
        "email": "a@x.org",
        "phone": "9",
        "account_enabled": True,
    }


def test_ensure_profile_keeps_profile_name_over_account(er, monkeypatch):
    monkeypatch.setattr(er, "get_expert_profile", lambda uid: {"user_id": uid, "display_name": "Perfil"})
    monkeypatch.setattr(er, "get_expert_account_by_email", lambda uid: {"display_name": "Conta", "enabled": 0})
    out = er.ensure_expert_profile("u")
    assert out["display_name"] == "Perfil" and out["account_enabled"] is False


def test_ensure_profile_creates_missing_profile_from_account(er, monkeypatch):
    store = {}
    monkeypatch.setattr(er, "get_expert_profile", lambda uid: store.get(uid))
    monkeypatch.setattr(er, "get_expert_account_by_email", lambda uid: {"display_name": "Bia", "enabled": 1})

    def upsert(uid, *, display_name, theme_ids):
        store[uid] = {"user_id": uid, "display_name": display_name, "theme_ids": theme_ids}

    monkeypatch.setattr(er, "upsert_expert_profile", upsert)
    out = er.ensure_expert_profile("b@x.org")
    assert store["b@x.org"]["display_name"] == "Bia"
    assert out["display_name"] == "Bia" and out["account_enabled"] is True


def test_ensure_profile_without_account_falls_back_to_user_id(er, monkeypatch):
    calls = []
    monkeypatch.setattr(er, "get_expert_profile", lambda uid: None)  # upsert não persistiu
    monkeypatch.setattr(er, "get_expert_account_by_email", lambda uid: None)
    monkeypatch.setattr(er, "upsert_expert_profile", lambda uid, **kw: calls.append((uid, kw)))
    out = er.ensure_expert_profile("svc")
    assert calls == [("svc", {"display_name": "svc", "theme_ids": []})]
    assert out == {"user_id": "svc", "theme_ids": [], "display_name": "svc"}


def test_update_profile_forwards_fields_and_rereads(er, monkeypatch):
    seen = {}
    monkeypatch.setattr(er, "upsert_expert_profile", lambda uid, **kw: seen.update(uid=uid, **kw))
    monkeypatch.setattr(er, "ensure_expert_profile", lambda uid: {"user_id": uid, "ok": True})
    out = er.update_expert_profile("u1", display_name="N", theme_ids=["a"], credentials_note="PhD")
    assert seen == {"uid": "u1", "display_name": "N", "theme_ids": ["a"], "credentials_note": "PhD"}
    assert out == {"user_id": "u1", "ok": True}


def test_catalog_pool_without_reference_requirement_keeps_all(er, monkeypatch):
    rows = [{"id": " h1 ", "query": "Q", "tags": None}, {"id": "h2", "theme": "fis", "reference": "R", "tags": ["x"]}]
    monkeypatch.setattr(er, "load_catalog_entries", lambda theme=None: rows)
    monkeypatch.setattr(er, "filter_entries_by_split", lambda r, split, seed=None: r)
    monkeypatch.setattr(er, "resolve_catalog_split", lambda eid, seed=None: f"split-{eid}")
    pool = er._catalog_review_pool(["hist"], require_reference=False)
    assert [(p["benchmark_id"], p["theme"], p["split"], p["tags"]) for p in pool] == [
        ("h1", "hist", "split-h1", []),
        ("h2", "fis", "split-h2", ["x"]),
    ]


def _eval_rows():
    return [
        {
            "prompt_text": "P1",
            "quality": "7.5",
            "model": "m",
            "metadata": {"benchmark_theme": "hist", "benchmark_id": "b1"},
        },
        {"prompt_text": "P2", "quality": None, "metadata": {"benchmark_theme": "fis", "benchmark_id": "b2"}},
        {"prompt_text": "P3", "metadata": {"benchmark_theme": "hist", "benchmark_id": "  "}},
        {"prompt_text": "P4", "metadata": None},
    ]


def test_eval_pool_filters_theme_and_blank_ids(er, monkeypatch):
    monkeypatch.setattr(er, "list_eval_run_results", lambda run_id, limit: _eval_rows())
    pool = er._eval_review_pool("run1", ["hist"])
    assert len(pool) == 1
    assert pool[0]["benchmark_id"] == "b1" and pool[0]["judge_quality"] == 7.5 and pool[0]["eval_run_id"] == "run1"
    unfiltered = er._eval_review_pool("run1", [])
    assert [p["benchmark_id"] for p in unfiltered] == ["b1", "b2"]
    assert unfiltered[1]["judge_quality"] == 0.0


def test_next_item_from_eval_run_skips_assessed(er, monkeypatch):
    monkeypatch.setattr(er, "ensure_expert_profile", lambda uid: {"theme_ids": ["hist", "fis", " "]})
    monkeypatch.setattr(er, "list_eval_run_results", lambda run_id, limit: _eval_rows())
    monkeypatch.setattr(er, "list_assessed_benchmark_ids", lambda uid, eval_run_id=None: ["b1"])
    item = er.get_next_review_item("e", eval_run_id="run1")
    assert item["benchmark_id"] == "b2" and item["source"] == "eval"

    monkeypatch.setattr(er, "list_assessed_benchmark_ids", lambda uid, eval_run_id=None: ["b1", "b2"])
    assert er.get_next_review_item("e", eval_run_id="run1") is None


def test_next_item_catalog_exhausted_returns_none(er, monkeypatch):
    monkeypatch.setattr(er, "ensure_expert_profile", lambda uid: {"theme_ids": ["hist"]})
    monkeypatch.setattr(er, "list_assessed_benchmark_ids", lambda *a, **k: ["h1", "h2"])
    monkeypatch.setattr(er, "_catalog_review_pool", lambda t, **k: [{"benchmark_id": "h1"}, {"benchmark_id": "h2"}])
    assert er.get_next_review_item("e", seed=3) is None


def test_submit_assessment_returns_latest_or_placeholder(er, monkeypatch):
    created = []
    monkeypatch.setattr(er, "create_expert_assessment", lambda **kw: created.append(kw))
    monkeypatch.setattr(er, "list_expert_assessments", lambda **kw: [{"id": 9, **kw}])
    kwargs = dict(benchmark_id="b", theme="t", query_text="q", answer="a", reference=None, eval_run_id=None)
    saved = er.submit_expert_assessment("e", judge_quality=None, quality_score=7.0, rubric={"x": 1}, **kwargs)
    assert saved["id"] == 9 and saved["expert_id"] == "e" and saved["limit"] == 1
    assert created[0]["expert_id"] == "e" and created[0]["quality_score"] == 7.0 and created[0]["notes"] is None

    monkeypatch.setattr(er, "list_expert_assessments", lambda **kw: [])
    assert er.submit_expert_assessment("e", judge_quality=1.0, quality_score=2.0, rubric={}, **kwargs) == {
        "status": "saved"
    }


def test_agreement_buckets_every_band_and_perfect_agreement(er, monkeypatch):
    scores = [1.0, 5.0, 7.0, 9.0]
    rows = [{"theme": None, "quality_score": s, "judge_quality": s + 0.5} for s in scores]
    rows.append({"theme": "x", "quality_score": 5.0, "judge_quality": None})  # sem par
    monkeypatch.setattr(er, "list_expert_assessments", lambda **kw: rows)
    report = er.expert_judge_agreement_report()
    assert report["pairs"] == 4
    assert report["kappa"] == pytest.approx(1.0)  # mesma faixa em todos os pares
    assert report["mean_absolute_error"] == pytest.approx(0.5)
    assert set(report["by_theme"]) == {"unknown"}
    assert report["delivery"]["method"] == "insufficient_pairs"


def test_delivery_agreement_inverts_scaffolding(er):
    rows = [
        {"rubric": {"scaffolding": 10}, "p_entrega": 0.0},  # andaime total = nenhuma entrega
        {"rubric": {"scaffolding": 0}, "p_entrega": 1.0},
        {"rubric": {"scaffolding": 5}, "p_entrega": 0.5},
        {"rubric": {"scaffolding": 5}, "p_entrega": None},
        {"rubric": "invalido", "p_entrega": 0.2},
    ]
    report = er.delivery_agreement_report(rows)
    assert report["pairs"] == 3
    assert report["mean_absolute_levels"] == 0
    assert report["kappa"] == pytest.approx(1.0)


def test_delivery_agreement_counts_level_distance(er):
    rows = [{"rubric": {"scaffolding": 10}, "p_entrega": 1.0}, {"rubric": {"scaffolding": 0}, "p_entrega": 1.0}]
    assert er.delivery_agreement_report(rows)["mean_absolute_levels"] == pytest.approx(2.0)


def test_list_available_themes_delegates(er, monkeypatch):
    monkeypatch.setattr(er, "list_themes_summary", lambda: [{"id": "t"}])
    assert er.list_available_themes() == [{"id": "t"}]


def test_kappa_dashboard_builds_theme_table(er, monkeypatch):
    report = {
        "kappa": 0.4,
        "pairs": 6,
        "mean_absolute_error": 1.2,
        "by_theme": {"zeta": {"kappa": 0.1, "n": 2}, "alfa": {"kappa": 0.9, "observed_agreement": 1.0}, "bad": 3},
    }
    monkeypatch.setattr(er, "expert_judge_agreement_report", lambda eval_run_id=None: report)
    monkeypatch.setattr(er, "list_themes_summary", lambda: [{"id": "alfa", "title": "Alfa"}, {"id": "zeta"}])
    monkeypatch.setattr("app.roadmap_features.get_expert_assessment_stats", lambda: {"total": "5", "experts": None})
    out = er.build_expert_kappa_dashboard(eval_run_id="r1")
    assert [row["theme_id"] for row in out["by_theme"]] == ["alfa", "zeta"]
    assert out["by_theme"][0]["theme_title"] == "Alfa" and out["by_theme"][0]["n"] == 0
    assert out["by_theme"][1]["theme_title"] == "zeta"
    assert (out["global_kappa"], out["global_n"], out["total_assessments"], out["active_experts"]) == (0.4, 6, 5, 0)
    assert out["insufficient_data"] is False and out["eval_run_id"] == "r1"


def test_kappa_dashboard_insufficient_data(er, monkeypatch):
    monkeypatch.setattr(er, "expert_judge_agreement_report", lambda eval_run_id=None: {"kappa": None, "n": 1})
    monkeypatch.setattr(er, "list_themes_summary", lambda: [])
    monkeypatch.setattr("app.roadmap_features.get_expert_assessment_stats", lambda: {})
    out = er.build_expert_kappa_dashboard()
    assert out["insufficient_data"] is True and out["global_n"] == 1 and out["by_theme"] == []
