# Objective: Tests for benchmark held-out split utilities.
"""Tests for deterministic catalog splits."""

from app.benchmark_splits import filter_entries_by_split, resolve_catalog_split, split_summary


def test_resolve_catalog_split_is_deterministic():
    a = resolve_catalog_split("hist_001", seed=42)
    b = resolve_catalog_split("hist_001", seed=42)
    assert a == b
    assert a in {"dev", "held_out"}


def test_filter_entries_by_split():
    rows = [{"id": f"item_{i}", "query": f"q{i}"} for i in range(100)]
    held = filter_entries_by_split(rows, "held_out", seed=42)
    dev = filter_entries_by_split(rows, "dev", seed=42)
    assert len(held) + len(dev) == 100
    assert all(resolve_catalog_split(r["id"], seed=42) == "held_out" for r in held)


def test_split_summary_counts():
    rows = [{"id": f"x{i}"} for i in range(50)]
    summary = split_summary(rows, seed=7)
    assert summary["dev"] + summary["held_out"] == 50
