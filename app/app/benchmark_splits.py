# Objective: Deterministic dev vs held-out splits for benchmark catalog entries.
"""Held-out split utilities for reproducible academic evaluation."""

from __future__ import annotations

import hashlib
import os
from typing import Literal, Optional

SplitName = Literal["dev", "held_out"]

DEFAULT_SPLIT_SEED = 42
DEFAULT_HELD_OUT_RATIO = 0.20


def _split_seed() -> int:
    raw = os.environ.get("BENCHMARK_SPLIT_SEED", str(DEFAULT_SPLIT_SEED)).strip()
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_SPLIT_SEED


def _held_out_ratio() -> float:
    raw = os.environ.get("BENCHMARK_HELD_OUT_RATIO", str(DEFAULT_HELD_OUT_RATIO)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = DEFAULT_HELD_OUT_RATIO
    return max(0.0, min(0.5, value))


def resolve_catalog_split(
    entry_id: str,
    *,
    seed: Optional[int] = None,
    held_out_ratio: Optional[float] = None,
) -> SplitName:
    """Assign one catalog entry to dev or held_out deterministically."""
    entry = str(entry_id or "").strip()
    if not entry:
        return "dev"
    seed_val = _split_seed() if seed is None else int(seed)
    ratio = _held_out_ratio() if held_out_ratio is None else float(held_out_ratio)
    digest = hashlib.sha256(f"{seed_val}:{entry}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10_000 / 10_000.0
    return "held_out" if bucket < ratio else "dev"


def filter_entries_by_split(
    rows: list[dict],
    split: Optional[str],
    *,
    seed: Optional[int] = None,
    held_out_ratio: Optional[float] = None,
) -> list[dict]:
    """Filter catalog rows by split name; returns all rows when split is unset."""
    wanted = (split or os.environ.get("BENCHMARK_SPLIT", "")).strip().lower()
    if wanted not in {"dev", "held_out"}:
        return list(rows)
    out: list[dict] = []
    for row in rows:
        entry_id = str(row.get("id") or row.get("query", "")).strip()
        if resolve_catalog_split(entry_id, seed=seed, held_out_ratio=held_out_ratio) == wanted:
            out.append(row)
    return out


def split_summary(rows: list[dict], *, seed: Optional[int] = None) -> dict[str, int]:
    """Count dev vs held_out assignments for a row list."""
    counts = {"dev": 0, "held_out": 0}
    for row in rows:
        entry_id = str(row.get("id") or row.get("query", "")).strip()
        split = resolve_catalog_split(entry_id, seed=seed)
        counts[split] += 1
    return counts
