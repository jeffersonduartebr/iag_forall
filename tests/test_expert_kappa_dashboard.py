# Objective: Tests for expert kappa dashboard builder.
"""Tests for kappa dashboard aggregation."""

from app.services import expert_review as er


def test_build_expert_kappa_dashboard_empty(monkeypatch):
    monkeypatch.setattr(er, "expert_judge_agreement_report", lambda **kwargs: {"kappa": None, "n": 0, "by_theme": {}})
    monkeypatch.setattr("app.roadmap_features.get_expert_assessment_stats", lambda: {"total": 0, "experts": 0, "themes": 0, "mae": 0.0})
    monkeypatch.setattr(er, "list_themes_summary", lambda: [{"id": "historia", "title": "História"}])
    out = er.build_expert_kappa_dashboard()
    assert out["insufficient_data"] is True
    assert out["by_theme"] == []


def test_build_expert_kappa_dashboard_with_themes(monkeypatch):
    monkeypatch.setattr(
        er,
        "expert_judge_agreement_report",
        lambda **kwargs: {
            "kappa": 0.72,
            "pairs": 5,
            "mean_absolute_error": 0.8,
            "by_theme": {
                "historia": {"kappa": 0.8, "n": 3, "observed_agreement": 0.9},
                "fisica": {"kappa": 0.5, "n": 2, "observed_agreement": 0.7},
            },
        },
    )
    monkeypatch.setattr(
        "app.roadmap_features.get_expert_assessment_stats",
        lambda: {"total": 5, "experts": 2, "themes": 2, "mae": 0.8},
    )
    monkeypatch.setattr(
        er,
        "list_themes_summary",
        lambda: [
            {"id": "historia", "title": "História"},
            {"id": "fisica", "title": "Física"},
        ],
    )
    out = er.build_expert_kappa_dashboard()
    assert out["global_kappa"] == 0.72
    assert len(out["by_theme"]) == 2
    assert out["by_theme"][0]["theme_title"] == "Física"  # sorted alphabetically by theme_id
