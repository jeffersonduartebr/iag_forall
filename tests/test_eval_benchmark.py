# Objective: Tests for benchmark-backed eval run resolution.
"""Tests for eval benchmark integration."""

from __future__ import annotations

import pytest
from app.eval_benchmark import list_benchmark_themes, resolve_eval_prompt_bundle
from app.schemas import EvalRunCreateRequest


def test_list_benchmark_themes_includes_adversarial():
    themes = {item["id"] for item in list_benchmark_themes()}
    assert "adversarial" in themes
    assert "multimodal" in themes


def test_resolve_eval_prompt_bundle_from_theme():
    payload = EvalRunCreateRequest(benchmark_theme="adversarial", benchmark_sample_size=5, benchmark_seed=3)
    prompts, catalog, meta = resolve_eval_prompt_bundle(payload)
    assert len(prompts) == 5
    assert len(catalog) == 5
    assert meta["benchmark_themes"] == ["adversarial"]
    assert all(item.get("attack_strategy") for item in catalog)


def test_resolve_eval_prompt_bundle_requires_rows():
    with pytest.raises(KeyError):
        resolve_eval_prompt_bundle(EvalRunCreateRequest(benchmark_theme="missing_theme_xyz"))
