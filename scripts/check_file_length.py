#!/usr/bin/env python3
"""Enforce a maximum SLOC-per-file limit with a grandfathered (ratchet) baseline.

SLOC = **logical** code lines: one per statement (``tokenize.NEWLINE``), counted
from the file as it is on disk. Blank lines, comment-only lines, and continuation
lines (a statement wrapped across several physical lines, inside brackets or a
multi-line string) do not add to the count. This makes the metric **invariant to
``ruff format``**, which only ever changes *physical* wrapping — so the two
pre-commit hooks (``ruff-format`` and this one) can never disagree. The check is
pure Python (no ruff needed at check time → deterministic in any environment).

Baselines are recorded on the ruff-**formatted** logical count (see ``--init`` /
``--update``): since formatting only ever *splits* compound one-liners (``if x: y``)
into more logical lines and never merges them, the on-disk logical count is always
``<=`` the recorded ceiling, so a later ``ruff format`` can never push a file over.
Recording the baseline needs ruff (a developer action); it falls back to the
on-disk logical count if ruff is unavailable.

Scope: ``app/app/**/*.py`` and ``tests/**/*.py``.

Modes:
- (default)   Fail if any in-scope file exceeds its allowed SLOC. The allowance is
              ``sloc_baseline.json[path]`` when present, else ``MAX_SLOC``.
- --init      One-time bootstrap: (re)write the baseline as ``{path: sloc}`` for
              every file currently over ``MAX_SLOC`` (formatted logical count).
- --update    Ratchet DOWN: lower each baseline ceiling to the current SLOC (never
              raise it), drop entries that fell to <= ``MAX_SLOC``, and refuse to
              grandfather a NEW file over the limit (it must be refactored).
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tokenize
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts" / "sloc_baseline.json"
MAX_SLOC = 500
SCAN_ROOTS = [ROOT / "app" / "app", ROOT / "tests"]
EXCLUDED_DIR_NAMES = {"__pycache__"}

_RUFF_CMD: Optional[List[str]] = None


def _logical_sloc(text: str) -> int:
    """Count logical code lines (one ``NEWLINE`` token per statement).

    Falls back to physical non-blank/non-comment lines if the source cannot be
    tokenized (e.g. a syntax error mid-edit)."""
    try:
        return sum(1 for tok in tokenize.generate_tokens(io.StringIO(text).readline) if tok.type == tokenize.NEWLINE)
    except Exception:
        return sum(1 for raw in text.splitlines() if raw.strip() and not raw.strip().startswith("#"))


def count_sloc(path: Path) -> int:
    """Logical SLOC of the file as it is on disk (formatting-invariant, pure Python)."""
    return _logical_sloc(path.read_text(encoding="utf-8", errors="replace"))


def _ruff_cmd() -> Optional[List[str]]:
    """Resolve a runnable ruff invocation once (PATH binary, else ``python -m ruff``)."""
    global _RUFF_CMD
    if _RUFF_CMD is not None:
        return _RUFF_CMD or None
    on_path = shutil.which("ruff")
    if on_path:
        _RUFF_CMD = [on_path]
        return _RUFF_CMD
    try:
        probe = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, timeout=30)
        if probe.returncode == 0:
            _RUFF_CMD = [sys.executable, "-m", "ruff"]
            return _RUFF_CMD
    except Exception:
        pass
    _RUFF_CMD = []  # sentinel: ruff unavailable
    return None


def baseline_sloc(path: Path) -> int:
    """Logical SLOC of the ruff-formatted file, used only when writing the baseline.

    Records the formatted count so a later ``ruff format`` can never exceed the
    ceiling. Falls back to the on-disk logical count when ruff is unavailable."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    cmd = _ruff_cmd()
    if not cmd:
        return _logical_sloc(raw)
    try:
        proc = subprocess.run(
            [*cmd, "format", "--stdin-filename", str(path), "-"],
            input=raw,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0 and proc.stdout:
            return _logical_sloc(proc.stdout)
    except Exception:
        pass
    return _logical_sloc(raw)


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


def compute_counts(counter=count_sloc) -> Dict[str, int]:
    """Return {relative_path: sloc} for every in-scope file using ``counter``.

    ``count_sloc`` (on-disk logical) drives the check; ``baseline_sloc`` (formatted
    logical) is used when writing the baseline."""
    return {_rel(path): counter(path) for path in iter_python_files()}


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
        print(f"OK: {len(counts)} files within the {MAX_SLOC}-SLOC limit ({len(baseline)} grandfathered).")
        return 0

    print(f"SLOC limit violations (max {MAX_SLOC} SLOC per file):")
    for path, sloc, allowed in sorted(violations, key=lambda item: -item[1]):
        reason = (
            "exceeds frozen baseline (file grew)" if allowed != MAX_SLOC else f"over {MAX_SLOC} (new or grown file)"
        )
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
    if "--init" in argv:
        return run_init(compute_counts(baseline_sloc))
    if "--update" in argv:
        return run_update(compute_counts(baseline_sloc), load_baseline())
    return run_check(compute_counts(count_sloc), load_baseline())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
