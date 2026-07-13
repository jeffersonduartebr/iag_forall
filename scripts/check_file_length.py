#!/usr/bin/env python3
"""Enforce a maximum SLOC-per-file limit with a grandfathered (ratchet) baseline.

SLOC = physical lines that are non-blank and not comment-only (a line whose
stripped form starts with ``#``). Docstrings count as code (line-based counting,
no AST — deterministic and simple).

Scope: ``app/app/**/*.py`` and ``tests/**/*.py``.

Modes:
- (default)   Fail if any in-scope file exceeds its allowed SLOC. The allowance is
              ``sloc_baseline.json[path]`` when present, else ``MAX_SLOC``.
- --init      One-time bootstrap: (re)write the baseline as ``{path: sloc}`` for
              every file currently over ``MAX_SLOC``.
- --update    Ratchet DOWN: lower each baseline ceiling to the current SLOC (never
              raise it), drop entries that fell to <= ``MAX_SLOC``, and refuse to
              grandfather a NEW file over the limit (it must be refactored).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts" / "sloc_baseline.json"
MAX_SLOC = 500
SCAN_ROOTS = [ROOT / "app" / "app", ROOT / "tests"]
EXCLUDED_DIR_NAMES = {"__pycache__"}


def count_sloc(path: Path) -> int:
    """Count non-blank, non-comment physical lines in one file."""
    total = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            total += 1
    return total


def iter_python_files() -> Iterable[Path]:
    """Yield in-scope Python files under the scan roots."""
    seen = set()
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            # Compare exclusions relative to the scan root so a repo checked out
            # under a directory named e.g. "__pycache__" is not fully skipped.
            if any(part in EXCLUDED_DIR_NAMES for part in path.relative_to(root).parts):
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def compute_counts() -> Dict[str, int]:
    """Return {relative_path: sloc} for every in-scope file."""
    return {_rel(path): count_sloc(path) for path in iter_python_files()}


def load_baseline() -> Dict[str, int]:
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {}


def _write_baseline(data: Dict[str, int]) -> None:
    ordered = dict(sorted(data.items()))
    BASELINE_PATH.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_check(counts: Dict[str, int], baseline: Dict[str, int]) -> int:
    """Fail when any file exceeds its allowance (baseline ceiling or MAX_SLOC)."""
    violations: List[Tuple[str, int, int]] = []
    for path, sloc in counts.items():
        allowed = baseline.get(path, MAX_SLOC)
        if sloc > allowed:
            violations.append((path, sloc, allowed))

    if not violations:
        print(f"OK: {len(counts)} files within the {MAX_SLOC}-SLOC limit "
              f"({len(baseline)} grandfathered).")
        return 0

    print(f"SLOC limit violations (max {MAX_SLOC} SLOC per file):")
    for path, sloc, allowed in sorted(violations, key=lambda item: -item[1]):
        reason = "exceeds frozen baseline (file grew)" if allowed != MAX_SLOC else f"over {MAX_SLOC} (new or grown file)"
        print(f"  {path}: {sloc} SLOC (allowed {allowed}) — {reason}")
    print(
        "\nRefactor the file(s) below the limit. If you legitimately SHRANK a "
        "baselined file, lower its ceiling with:\n"
        "    python3 scripts/check_file_length.py --update"
    )
    return 1


def run_init(counts: Dict[str, int]) -> int:
    """Bootstrap the baseline from the current set of over-limit files."""
    baseline = {path: sloc for path, sloc in counts.items() if sloc > MAX_SLOC}
    _write_baseline(baseline)
    print(f"Initialized {BASELINE_PATH.relative_to(ROOT)} with {len(baseline)} grandfathered file(s).")
    return 0


def run_update(counts: Dict[str, int], baseline: Dict[str, int]) -> int:
    """Ratchet baseline ceilings down; never raise; refuse new over-limit files."""
    new_baseline: Dict[str, int] = {}
    for path, sloc in counts.items():
        if sloc <= MAX_SLOC:
            continue
        ceiling = baseline.get(path)
        if ceiling is None:
            print(
                f"ERROR: {path} has {sloc} SLOC (> {MAX_SLOC}) and is not baselined.\n"
                "New/grown files cannot be grandfathered via --update; refactor it below the limit."
            )
            return 1
        new_baseline[path] = min(ceiling, sloc)
    _write_baseline(new_baseline)
    print(f"Ratcheted baseline: {len(new_baseline)} grandfathered file(s).")
    return 0


def main(argv: List[str]) -> int:
    counts = compute_counts()
    if "--init" in argv:
        return run_init(counts)
    if "--update" in argv:
        return run_update(counts, load_baseline())
    return run_check(counts, load_baseline())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
