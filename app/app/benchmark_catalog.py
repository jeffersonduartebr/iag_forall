# Objective: Load benchmark query catalog from JSONL for API, eval, and load tests.
"""Benchmark query catalog loader."""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - optional at runtime in minimal envs
    yaml = None

VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard", "complex", "expert"})
VALID_LANGS = frozenset({"pt", "en", "mixed"})
OPTIONAL_CATALOG_FIELDS = frozenset({"reference", "attack_strategy", "image_path", "criticality"})

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CATALOG_DIR = _REPO_ROOT / "data" / "benchmark_queries"


def resolve_catalog_dir() -> Path:
    """Return catalog directory from env or default repo path."""
    env = os.environ.get("BENCHMARK_CATALOG_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_CATALOG_DIR.resolve()


@lru_cache(maxsize=1)
def load_manifest() -> Dict[str, Any]:
    """Load and cache manifest.yaml."""
    catalog_dir = resolve_catalog_dir()
    manifest_path = catalog_dir / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load benchmark manifest")
    with manifest_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "themes" not in data:
        raise ValueError("Invalid benchmark manifest: missing themes")
    return data


def list_theme_ids() -> List[str]:
    """Return ordered theme ids from manifest."""
    manifest = load_manifest()
    return [str(t["id"]) for t in manifest["themes"]]


def list_themes_summary() -> List[Dict[str, Any]]:
    """Return manifest theme metadata for admin/eval APIs."""
    manifest = load_manifest()
    return [
        {
            "id": str(theme["id"]),
            "title": str(theme.get("title", theme["id"])),
            "target_count": get_theme_target_count(theme, manifest),
            "file": str(theme.get("file", "")),
            "subtopics": list(theme.get("subtopics") or []),
        }
        for theme in manifest["themes"]
    ]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be object at {path}:{line_no}")
            rows.append(row)
    return rows


def load_theme(theme_id: str) -> List[Dict[str, Any]]:
    """Load all queries for one theme."""
    manifest = load_manifest()
    catalog_dir = resolve_catalog_dir()
    theme = next((t for t in manifest["themes"] if t["id"] == theme_id), None)
    if theme is None:
        raise KeyError(f"Unknown benchmark theme: {theme_id}")
    path = catalog_dir / str(theme["file"])
    return _read_jsonl(path)


@lru_cache(maxsize=1)
def load_all_catalog_entries() -> tuple[Dict[str, Any], ...]:
    """Load every catalog entry as immutable tuple for caching."""
    entries: List[Dict[str, Any]] = []
    for theme_id in list_theme_ids():
        entries.extend(load_theme(theme_id))
    return tuple(entries)


def load_catalog_entries(
    *,
    theme: Optional[str] = None,
    difficulty: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load catalog entries with optional filters."""
    theme_filter = (theme or os.environ.get("BENCHMARK_THEME", "")).strip() or None
    diff_filter = (difficulty or os.environ.get("BENCHMARK_DIFFICULTY", "")).strip() or None

    if theme_filter:
        rows = list(load_theme(theme_filter))
    else:
        rows = [dict(r) for r in load_all_catalog_entries()]

    if diff_filter:
        rows = [r for r in rows if str(r.get("difficulty", "")) == diff_filter]
    if tags:
        wanted = {str(t).strip() for t in tags if str(t).strip()}
        if wanted:
            rows = [r for r in rows if wanted.intersection({str(x) for x in (r.get("tags") or [])})]
    return rows


def sample_catalog_entries(
    *,
    themes: Optional[List[str]] = None,
    difficulty: Optional[str] = None,
    tags: Optional[List[str]] = None,
    sample_size: int = 50,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return a deterministic or random sample from the catalog."""
    rows: List[Dict[str, Any]] = []
    theme_list = [str(t).strip() for t in (themes or []) if str(t).strip()]
    if theme_list:
        for theme_id in theme_list:
            rows.extend(load_catalog_entries(theme=theme_id, difficulty=difficulty, tags=tags))
    else:
        rows = load_catalog_entries(difficulty=difficulty, tags=tags)
    if sample_size <= 0 or len(rows) <= sample_size:
        return rows
    rng = random.Random(seed)
    picked = rows.copy()
    rng.shuffle(picked)
    return picked[:sample_size]


def load_all_queries() -> List[Dict[str, str]]:
    """Return Locust-compatible list of {\"query\": ...} dicts."""
    return [{"query": str(row["query"])} for row in load_catalog_entries() if row.get("query")]


def load_queries() -> List[Dict[str, str]]:
    """Alias for Locust backward compatibility."""
    return load_all_queries()


def get_theme_meta(theme_id: str) -> Dict[str, Any]:
    """Return manifest metadata for a theme."""
    manifest = load_manifest()
    theme = next((t for t in manifest["themes"] if t["id"] == theme_id), None)
    if theme is None:
        raise KeyError(f"Unknown benchmark theme: {theme_id}")
    return dict(theme)


def get_theme_target_count(theme: Dict[str, Any], manifest: Optional[Dict[str, Any]] = None) -> int:
    """Return expected query count for a theme (per-theme override or manifest default)."""
    if theme.get("target_count") is not None:
        return int(theme["target_count"])
    if manifest is None:
        manifest = load_manifest()
    return int(manifest.get("target_count", 150))


def get_theme_target_count_by_id(theme_id: str) -> int:
    """Return expected query count for a theme id."""
    return get_theme_target_count(get_theme_meta(theme_id))


def get_expected_total_queries() -> int:
    """Return total expected queries across all manifest themes."""
    manifest = load_manifest()
    return sum(get_theme_target_count(theme, manifest) for theme in manifest["themes"])


def load_programming_challenges() -> List[Dict[str, str]]:
    """Return Locust-compatible queries from programacao_desafios theme only."""
    return [
        {"query": str(row["query"])}
        for row in load_theme("programacao_desafios")
        if row.get("query")
    ]


def append_catalog_entry(theme_id: str, entry: Dict[str, Any]) -> Path:
    """Append one curated entry to a theme JSONL file (used by adversarial suite)."""
    manifest = load_manifest()
    theme = next((t for t in manifest["themes"] if t["id"] == theme_id), None)
    if theme is None:
        raise KeyError(f"Unknown benchmark theme: {theme_id}")
    path = resolve_catalog_dir() / str(theme["file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    load_manifest.cache_clear()
    load_all_catalog_entries.cache_clear()
    return path


def resolve_catalog_asset_path(path_value: str) -> Path:
    """Resolve image/asset path relative to repo root or catalog dir."""
    raw = Path(str(path_value))
    if raw.is_file():
        return raw.resolve()
    repo_candidate = _REPO_ROOT / raw
    if repo_candidate.is_file():
        return repo_candidate.resolve()
    catalog_candidate = resolve_catalog_dir() / raw
    if catalog_candidate.is_file():
        return catalog_candidate.resolve()
    raise FileNotFoundError(f"Catalog asset not found: {path_value}")
