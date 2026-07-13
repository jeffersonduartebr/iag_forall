#!/usr/bin/env python3
"""Export benchmark catalog release manifest for Zenodo / academic publication."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))

from app.benchmark_catalog import (  # noqa: E402
    get_expected_total_queries,
    list_theme_ids,
    load_all_catalog_entries,
    load_manifest,
    resolve_catalog_dir,
)
from app.benchmark_splits import split_summary  # noqa: E402


def main() -> int:
    catalog_dir = resolve_catalog_dir()
    manifest = load_manifest()
    rows = [dict(r) for r in load_all_catalog_entries()]
    splits = split_summary(rows)

    hasher = hashlib.sha256()
    for theme in manifest.get("themes") or []:
        path = catalog_dir / str(theme.get("file", ""))
        if path.is_file():
            hasher.update(path.read_bytes())

    release = {
        "schema": "iag-benchmark-release/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_dir": str(catalog_dir),
        "manifest_version": manifest.get("version"),
        "theme_count": len(list_theme_ids()),
        "total_queries": len(rows),
        "expected_total_queries": get_expected_total_queries(),
        "content_sha256": hasher.hexdigest(),
        "split_counts": splits,
        "themes": [
            {
                "id": t["id"],
                "title": t.get("title"),
                "file": t.get("file"),
                "target_count": t.get("target_count"),
            }
            for t in manifest.get("themes") or []
        ],
        "citation": "IAG ForAll Benchmark Catalog — multi-domain academic evaluation set",
        "license": "See repository LICENSE",
    }

    out_dir = REPO_ROOT / "data" / "benchmark_queries" / "releases"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = out_dir / f"benchmark_release_{stamp}.json"
    out_path.write_text(json.dumps(release, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RELEASE: {out_path}")
    print(f"QUERIES: {release['total_queries']} themes={release['theme_count']} splits={splits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
