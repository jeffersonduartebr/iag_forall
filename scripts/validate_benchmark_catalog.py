#!/usr/bin/env python3
"""Validate benchmark query catalog integrity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "data" / "benchmark_queries"
MANIFEST_PATH = CATALOG_DIR / "manifest.yaml"

VALID_DIFFICULTIES = {"easy", "medium", "hard", "complex", "expert"}
VALID_LANGS = {"pt", "en", "mixed"}


def theme_target_count(theme: dict, manifest: dict) -> int:
    if theme.get("target_count") is not None:
        return int(theme["target_count"])
    return int(manifest.get("target_count", 150))


def main() -> int:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    default_target = int(manifest.get("target_count", 150))
    themes = manifest.get("themes") or []
    if len(themes) != 34:
        print(f"ERROR: expected 34 themes, found {len(themes)}")
        return 1

    all_ids: set[str] = set()
    total = 0
    for theme in themes:
        theme_id = str(theme["id"])
        expected = theme_target_count(theme, manifest)
        path = CATALOG_DIR / str(theme["file"])
        if not path.is_file():
            print(f"ERROR: missing file for {theme_id}: {path}")
            return 1
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != expected:
            print(f"ERROR: {theme_id} has {len(rows)} rows, expected {expected}")
            return 1
        for row in rows:
            rid = str(row.get("id", ""))
            if rid in all_ids:
                print(f"ERROR: duplicate id {rid}")
                return 1
            all_ids.add(rid)
            if not str(row.get("query", "")).strip():
                print(f"ERROR: empty query in {rid}")
                return 1
            if str(row.get("theme")) != theme_id:
                print(f"ERROR: theme mismatch in {rid}")
                return 1
            if str(row.get("difficulty")) not in VALID_DIFFICULTIES:
                print(f"ERROR: invalid difficulty in {rid}")
                return 1
            if str(row.get("lang")) not in VALID_LANGS:
                print(f"ERROR: invalid lang in {rid}")
                return 1
            for optional_field in ("reference", "attack_strategy", "image_path", "criticality"):
                if optional_field in row and row[optional_field] is not None and not str(row[optional_field]).strip():
                    print(f"ERROR: empty optional field {optional_field} in {rid}")
                    return 1
        total += len(rows)
        print(f"OK {theme_id}: {len(rows)}")

    expected_total = sum(theme_target_count(t, manifest) for t in themes)
    if total != expected_total:
        print(f"ERROR: total {total}, expected {expected_total}")
        return 1

    print(f"VALID: {total} queries across {len(themes)} themes (default target {default_target})")
    try:
        sys.path.insert(0, str(ROOT / "app"))
        from app.benchmark_catalog import load_all_catalog_entries
        from app.benchmark_splits import split_summary

        rows = [dict(r) for r in load_all_catalog_entries()]
        splits = split_summary(rows)
        print(f"SPLITS: dev={splits['dev']} held_out={splits['held_out']}")
    except Exception as exc:
        print(f"SPLITS: skipped ({exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
