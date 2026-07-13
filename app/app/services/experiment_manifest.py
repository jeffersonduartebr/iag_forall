# Objective: Build reproducible experiment manifests for academic eval runs.
"""Experiment manifest generation with git hash and config snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.benchmark_catalog import (
    get_expected_total_queries,
    load_all_catalog_entries,
    load_manifest,
    resolve_catalog_dir,
)
from app.benchmark_splits import DEFAULT_HELD_OUT_RATIO, DEFAULT_SPLIT_SEED, split_summary


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_info() -> Dict[str, Any]:
    root = _repo_root()
    info: Dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["commit"] = commit
    except Exception:
        info["commit"] = os.environ.get("GIT_COMMIT", "unknown")

    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["dirty"] = bool(dirty)
    except Exception:
        info["dirty"] = None

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["branch"] = branch
    except Exception:
        info["branch"] = None
    return info


def _catalog_fingerprint() -> str:
    catalog_dir = resolve_catalog_dir()
    hasher = hashlib.sha256()
    manifest_path = catalog_dir / "manifest.yaml"
    if manifest_path.is_file():
        hasher.update(manifest_path.read_bytes())
    for theme in load_manifest().get("themes") or []:
        theme_path = catalog_dir / str(theme.get("file", ""))
        if theme_path.is_file():
            hasher.update(theme_path.name.encode("utf-8"))
            hasher.update(theme_path.read_bytes())
    return hasher.hexdigest()


def _config_snapshot() -> Dict[str, Any]:
    from app.settings_dynamic import settings

    keys = [
        "NSGA_W_QUALITY",
        "NSGA_W_LATENCY",
        "NSGA_W_COST",
        "BANDIT_EPSILON",
        "UNCERTAINTY_THRESHOLD",
        "CANDIDATE_MODELS_LIST",
        "JUDGE_LLMS",
        "JUDGE_LLM_MODEL",
        "TEMPERATURE_DEFAULT",
        "MAX_TOKENS_DEFAULT",
        "OPENROUTER_EXPLORATION_ENABLED",
        "OPENROUTER_EXPLORATION_RATE",
    ]
    snapshot: Dict[str, Any] = {}
    for key in keys:
        try:
            snapshot[key] = settings.get(key, getattr(settings, key, None))
        except Exception:
            snapshot[key] = None
    return snapshot


def build_experiment_manifest(
    *,
    run_id: str,
    benchmark_seed: Optional[int] = None,
    benchmark_split: Optional[str] = None,
    frozen_policy: bool = False,
    benchmark_themes: Optional[list[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a reproducibility manifest for one eval run."""
    manifest_yaml = load_manifest()
    all_rows = [dict(r) for r in load_all_catalog_entries()]
    split_counts = split_summary(all_rows, seed=benchmark_seed or DEFAULT_SPLIT_SEED)

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": time.time(),
        "git": _git_info(),
        "benchmark": {
            "catalog_dir": str(resolve_catalog_dir()),
            "manifest_version": manifest_yaml.get("version"),
            "catalog_fingerprint_sha256": _catalog_fingerprint(),
            "expected_total_queries": get_expected_total_queries(),
            "split_seed": benchmark_seed if benchmark_seed is not None else DEFAULT_SPLIT_SEED,
            "held_out_ratio": DEFAULT_HELD_OUT_RATIO,
            "split_filter": benchmark_split,
            "split_counts": split_counts,
            "themes": benchmark_themes or [],
        },
        "frozen_policy": bool(frozen_policy),
        "config_snapshot": _config_snapshot(),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def write_experiment_manifest(run_id: str, manifest: Dict[str, Any], output_dir: Optional[Path] = None) -> Path:
    """Persist manifest JSON next to thesis results or a custom directory."""
    base = output_dir or (_repo_root() / "thesis_results" / "experiment_manifests")
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{run_id}_experiment_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
