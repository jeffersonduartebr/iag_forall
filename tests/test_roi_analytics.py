# Objective: Tests for ROI analytics service.
"""Tests for savings vs baseline calculations."""

from app.services import roi_analytics as roi


def test_estimate_tokens():
    assert roi._estimate_tokens("abcd") >= 1


def test_baseline_unit_cost_uses_registry(monkeypatch):
    class FakeCfg:
        def calculate_cost(self, inp, out):
            return 0.05

    monkeypatch.setattr(roi, "get_model_config", lambda name: FakeCfg())
    cost = roi._baseline_unit_cost(query_text="a" * 400, answer="b" * 400, modality="text", baseline_model="openai/gpt-4o")
    assert cost == 0.05


def test_build_roi_report_with_rows(monkeypatch):
    rows = [
        {
            "id": 1,
            "query_text": "pergunta " * 50,
            "answer": "resposta " * 80,
            "chosen_model": "ollama/gemma3:4b",
            "modality": "text",
            "quality": 7.5,
            "abstained": 0,
            "estimated_cost_usd": 0.001,
            "cost_per_1k": 0.001,
            "created_at": "2026-07-01 10:00:00",
        },
        {
            "id": 2,
            "query_text": "outra " * 40,
            "answer": "outra resposta " * 60,
            "chosen_model": "openai/gpt-4o-mini",
            "modality": "text",
            "quality": 8.0,
            "abstained": 0,
            "estimated_cost_usd": 0.01,
            "cost_per_1k": 0.01,
            "created_at": "2026-07-02 11:00:00",
        },
    ]
    monkeypatch.setattr(roi, "_load_query_rows", lambda **kwargs: rows)

    class FakeCfg:
        def calculate_cost(self, inp, out):
            return 0.10

    monkeypatch.setattr(roi, "get_model_config", lambda name: FakeCfg())
    report = roi.build_roi_report(days=30)
    assert report["insufficient_data"] is False
    assert report["summary"]["query_count"] == 2
    assert report["summary"]["savings_usd"] > 0
    assert len(report["daily_series"]) == 2
    assert report["summary"]["savings_pct"] > 0


def test_build_roi_report_empty(monkeypatch):
    monkeypatch.setattr(roi, "_load_query_rows", lambda **kwargs: [])
    monkeypatch.setattr(roi, "get_usage_summary", lambda tenant_id=None: {"tenants": {}})
    report = roi.build_roi_report(days=7)
    assert report["insufficient_data"] is True
