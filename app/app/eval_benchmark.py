# Objective: Helpers to resolve eval prompts from benchmark catalog.
"""Benchmark catalog integration for evaluation runs."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.benchmark_catalog import list_themes_summary, sample_catalog_entries
from app.benchmark_splits import filter_entries_by_split
from app.schemas import EvalRunCreateRequest


def resolve_eval_prompt_bundle(payload: EvalRunCreateRequest) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """Resolve prompts and catalog metadata from explicit prompts or benchmark filters."""
    catalog_meta: Dict[str, Any] = {}
    prompt_catalog: List[Dict[str, Any]] = []

    themes = [str(t).strip() for t in (payload.benchmark_themes or []) if str(t).strip()]
    if payload.benchmark_theme and payload.benchmark_theme not in themes:
        themes.insert(0, str(payload.benchmark_theme).strip())

    if themes:
        rows = sample_catalog_entries(
            themes=themes,
            difficulty=payload.benchmark_difficulty,
            tags=payload.benchmark_tags or None,
            sample_size=int(payload.benchmark_sample_size or 10_000),
            seed=payload.benchmark_seed,
        )
        rows = filter_entries_by_split(
            rows,
            payload.benchmark_split,
            seed=payload.benchmark_seed,
        )
        if not rows:
            raise ValueError("benchmark catalog filters returned no prompts")
        prompt_catalog = [
            {
                "id": row.get("id"),
                "theme": row.get("theme"),
                "query": row.get("query"),
                "reference": row.get("reference"),
                "attack_strategy": row.get("attack_strategy"),
                "image_path": row.get("image_path"),
                "tags": row.get("tags") or [],
                "criticality": row.get("criticality"),
            }
            for row in rows
        ]
        prompts = [str(row["query"]) for row in prompt_catalog if str(row.get("query", "")).strip()]
        catalog_meta = {
            "benchmark_themes": themes,
            "benchmark_sample_size": payload.benchmark_sample_size,
            "benchmark_tags": payload.benchmark_tags,
            "benchmark_difficulty": payload.benchmark_difficulty,
            "benchmark_seed": payload.benchmark_seed,
            "benchmark_split": payload.benchmark_split,
            "frozen_policy": bool(payload.frozen_policy),
            "prompt_catalog": prompt_catalog,
        }
        return prompts, prompt_catalog, catalog_meta

    prompts = [str(prompt) for prompt in (payload.prompts or []) if str(prompt).strip()]
    return prompts, prompt_catalog, catalog_meta


def list_benchmark_themes() -> List[Dict[str, Any]]:
    """Return catalog themes for eval admin APIs."""
    return list_themes_summary()
