# Objective: Tests for expert review service logic.
"""Tests for expert review queue and kappa without DB."""



from app.services import expert_review as er


def test_catalog_review_pool_filters_reference(monkeypatch):
    monkeypatch.setattr(
        er,
        "load_catalog_entries",
        lambda theme=None, **kwargs: [
            {"id": "h1", "theme": "historia", "query": "Q1", "reference": "R1"},
            {"id": "h2", "theme": "historia", "query": "Q2"},
        ],
    )
    pool = er._catalog_review_pool(["historia"], split=None, require_reference=True)
    assert len(pool) == 1
    assert pool[0]["benchmark_id"] == "h1"


def test_expert_judge_agreement_insufficient_pairs(monkeypatch):
    monkeypatch.setattr(er, "list_expert_assessments", lambda **kwargs: [])
    report = er.expert_judge_agreement_report()
    assert report["kappa"] is None


def test_expert_judge_agreement_with_pairs(monkeypatch):
    monkeypatch.setattr(
        er,
        "list_expert_assessments",
        lambda **kwargs: [
            {"theme": "historia", "quality_score": 8.0, "judge_quality": 7.5},
            {"theme": "historia", "quality_score": 6.0, "judge_quality": 6.5},
            {"theme": "fisica", "quality_score": 9.0, "judge_quality": 8.0},
        ],
    )
    report = er.expert_judge_agreement_report()
    assert report["pairs"] == 3
    assert report["kappa"] is not None


def test_get_next_review_item_requires_themes(monkeypatch):
    monkeypatch.setattr(er, "ensure_expert_profile", lambda uid: {"user_id": uid, "theme_ids": []})
    assert er.get_next_review_item("expert1") is None


def test_get_next_review_item_returns_catalog_item(monkeypatch):
    monkeypatch.setattr(er, "ensure_expert_profile", lambda uid: {"user_id": uid, "theme_ids": ["historia"]})
    monkeypatch.setattr(er, "list_assessed_benchmark_ids", lambda *a, **k: [])
    monkeypatch.setattr(
        er,
        "_catalog_review_pool",
        lambda themes, **kwargs: [
            {"benchmark_id": "h1", "theme": "historia", "query_text": "Q?", "reference": "R"},
        ],
    )
    item = er.get_next_review_item("expert1")
    assert item is not None
    assert item["benchmark_id"] == "h1"
