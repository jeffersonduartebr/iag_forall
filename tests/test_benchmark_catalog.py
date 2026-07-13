# Objective: Tests for benchmark query catalog loader and data integrity.
"""Tests for benchmark query catalog."""

from __future__ import annotations

import json
from pathlib import Path

import benchmark_catalog as bc
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = REPO_ROOT / "data" / "benchmark_queries"


@pytest.fixture(autouse=True)
def _reset_catalog_cache():
    bc.load_manifest.cache_clear()
    bc.load_all_catalog_entries.cache_clear()
    yield
    bc.load_manifest.cache_clear()
    bc.load_all_catalog_entries.cache_clear()


def test_manifest_has_34_themes():
    manifest = bc.load_manifest()
    assert len(manifest["themes"]) == 34
    assert int(manifest["target_count"]) == 150


@pytest.mark.parametrize("theme_id", bc.list_theme_ids())
def test_each_theme_has_expected_query_count(theme_id: str):
    rows = bc.load_theme(theme_id)
    assert len(rows) == bc.get_theme_target_count_by_id(theme_id)


def test_catalog_ids_are_unique_globally():
    ids = [str(row["id"]) for row in bc.load_all_catalog_entries()]
    assert len(ids) == len(set(ids))


def test_load_all_queries_locust_shape():
    queries = bc.load_all_queries()
    assert len(queries) == bc.get_expected_total_queries()
    assert all(set(item.keys()) == {"query"} for item in queries)
    assert all(str(item["query"]).strip() for item in queries)


def test_programming_challenges_theme():
    rows = bc.load_theme("programacao_desafios")
    assert len(rows) == 40
    assert all(row["difficulty"] == "expert" for row in rows)
    challenges = bc.load_programming_challenges()
    assert len(challenges) == 40


def test_adversarial_theme_has_attack_metadata():
    rows = bc.load_theme("adversarial")
    assert len(rows) == 60
    assert all(row.get("attack_strategy") for row in rows)
    assert all(row.get("reference") for row in rows)


def test_multimodal_theme_has_image_paths():
    rows = bc.load_theme("multimodal")
    assert len(rows) == 30
    assert all(row.get("image_path") for row in rows)


def test_sample_catalog_entries_respects_size():
    rows = bc.sample_catalog_entries(themes=["historia"], sample_size=10, seed=7)
    assert len(rows) == 10
    assert all(row["theme"] == "historia" for row in rows)


def test_theme_filter_env(monkeypatch):
    monkeypatch.setenv("BENCHMARK_THEME", "historia")
    rows = bc.load_catalog_entries()
    assert len(rows) == 150
    assert all(row["theme"] == "historia" for row in rows)


def test_difficulty_filter_env(monkeypatch):
    monkeypatch.setenv("BENCHMARK_DIFFICULTY", "complex")
    rows = bc.load_catalog_entries()
    assert rows
    assert all(row["difficulty"] == "complex" for row in rows)


def test_jsonl_files_exist_for_all_themes():
    manifest = bc.load_manifest()
    for theme in manifest["themes"]:
        theme_id = str(theme["id"])
        path = CATALOG_DIR / str(theme["file"])
        assert path.is_file(), theme_id
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == bc.get_theme_target_count(theme)
        first = json.loads(lines[0])
        assert first["theme"] == theme_id


def test_hard_and_complex_rows_have_reference_except_special_themes():
    skip = {"programacao_desafios", "adversarial", "multimodal"}
    for theme_id in bc.list_theme_ids():
        if theme_id in skip:
            continue
        for row in bc.load_theme(theme_id):
            if str(row.get("difficulty")) in {"hard", "complex"}:
                assert str(row.get("reference", "")).strip(), f"missing reference in {row.get('id')}"
