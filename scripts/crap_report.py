#!/usr/bin/env python3
"""Report and ratchet CRAP (Change Risk Anti-Patterns) scores per function.

CRAP(f) = CC(f)^2 * (1 - cov(f))^3 + CC(f)

- CC: cyclomatic complexity from ruff's mccabe rule (``C901``, threshold 0 so
  every function is reported).
- cov: statement + branch coverage of the function, read from the
  ``functions`` section of a coverage.py JSON report (coverage >= 7.6).

A function with CC 10 needs ~48% coverage to stay under 30; CC 25 needs ~81%.
High CRAP means "complex and untested": the riskiest code to change.

Typical use (from the repo root):

    pytest tests/ --cov=app/app --cov-branch --cov-report=json:coverage.json
    python3 scripts/crap_report.py                  # report + ratchet check
    python3 scripts/crap_report.py --update         # lower ceilings after improving
    python3 scripts/crap_report.py --init           # (re)bootstrap the baseline

Ratchet: ``scripts/crap_baseline.json`` maps ``path::qualname`` to a CRAP
ceiling for functions already above ``THRESHOLD``. The check fails when a
listed function grows past its ceiling or an unlisted function exceeds the
threshold. ``--update`` only lowers ceilings and drops fixed entries; it never
grandfathers new offenders. Files absent from the coverage report (not
measured) are listed separately and never gate.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts" / "crap_baseline.json"
SCAN_ROOT = "app/app"
THRESHOLD = 30.0
_CC_MESSAGE = re.compile(r"`([^`]+)` is too complex \((\d+) >")


class FunctionScore(NamedTuple):
    key: str  # "path::qualname"
    path: str
    line: int
    cc: int
    coverage: float  # 0..1
    crap: float


def crap_score(cc: int, coverage: float) -> float:
    """CRAP = CC² · (1 − cov)³ + CC."""
    return cc * cc * (1.0 - coverage) ** 3 + cc


def complexity_by_function(scan_root: str = SCAN_ROOT) -> List[Tuple[str, int, str, int]]:
    """Return ``(path, line, name, cc)`` for every function under ``scan_root`` via ruff."""
    cmd = [
        "ruff",
        "check",
        scan_root,
        "--select",
        "C901",
        "--config",
        "lint.mccabe.max-complexity=0",
        "--output-format",
        "json",
        "--no-cache",
        "--exit-zero",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    out: List[Tuple[str, int, str, int]] = []
    for item in json.loads(proc.stdout or "[]"):
        match = _CC_MESSAGE.search(item.get("message", ""))
        if not match:
            continue
        path = str(Path(item["filename"]).resolve().relative_to(ROOT))
        out.append((path, int(item["location"]["row"]), match.group(1), int(match.group(2))))
    return out


def _function_index(coverage_json: dict) -> Dict[Tuple[str, int], Tuple[str, float]]:
    """Map ``(path, start_line)`` to ``(qualname, coverage)`` from a coverage.py JSON report."""
    index: Dict[Tuple[str, int], Tuple[str, float]] = {}
    for path, data in coverage_json.get("files", {}).items():
        for qualname, fn in (data.get("functions") or {}).items():
            start = fn.get("start_line")
            if not qualname or not start:
                continue
            s = fn["summary"]
            total = s.get("num_statements", 0) + s.get("num_branches", 0)
            covered = s.get("covered_lines", 0) + s.get("covered_branches", 0)
            index[(path, int(start))] = (qualname, covered / total if total else 1.0)
    return index


def score_functions(coverage_json: dict) -> Tuple[List[FunctionScore], List[Tuple[str, int, str, int]]]:
    """Join complexity and coverage; returns ``(scores, unmeasured)``."""
    index = _function_index(coverage_json)
    measured_files = set(coverage_json.get("files", {}))
    scores: List[FunctionScore] = []
    unmeasured: List[Tuple[str, int, str, int]] = []
    for path, line, name, cc in complexity_by_function():
        hit: Optional[Tuple[str, float]] = None
        for delta in (0, -1, 1, -2, 2, -3, 3):  # decorators shift the recorded start line
            hit = index.get((path, line + delta))
            if hit:
                break
        if hit is None:
            if path not in measured_files:
                unmeasured.append((path, line, name, cc))
            continue
        qualname, cov = hit
        scores.append(FunctionScore(f"{path}::{qualname}", path, line, cc, cov, crap_score(cc, cov)))
    return scores, unmeasured


def load_baseline() -> Dict[str, float]:
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {}


def _write_baseline(data: Dict[str, float]) -> None:
    ordered = {k: round(v, 1) for k, v in sorted(data.items())}
    BASELINE_PATH.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_report(scores: List[FunctionScore], unmeasured: list, top: int) -> None:
    offenders = [s for s in scores if s.crap > THRESHOLD]
    print(f"CRAP report: {len(scores)} functions measured, {len(offenders)} above {THRESHOLD:g}.")
    for s in sorted(scores, key=lambda s: -s.crap)[:top]:
        print(
            f"  CRAP {s.crap:7.1f}  CC {s.cc:3d}  cov {s.coverage * 100:5.1f}%  {s.path}:{s.line}  {s.key.split('::')[1]}"
        )
    if unmeasured:
        print(f"  ({len(unmeasured)} functions in files absent from the coverage report were not scored)")


def run_check(scores: List[FunctionScore], baseline: Dict[str, float]) -> int:
    violations = []
    for s in scores:
        ceiling = baseline.get(s.key, THRESHOLD)
        if s.crap > ceiling + 1e-6:
            violations.append((s, ceiling))
    if not violations:
        print(f"OK: no function above its CRAP allowance ({len(baseline)} grandfathered).")
        return 0
    print("CRAP violations:")
    for s, ceiling in sorted(violations, key=lambda item: -item[0].crap):
        why = "grew past its baseline" if s.key in baseline else f"new function above {THRESHOLD:g}"
        print(f"  {s.key}: CRAP {s.crap:.1f} (allowed {ceiling:.1f}; CC {s.cc}, cov {s.coverage * 100:.0f}%) — {why}")
    print(
        "\nAdd tests (raise coverage) or split the function (lower CC). "
        "After improving a baselined function: python3 scripts/crap_report.py --update"
    )
    return 1


def run_update(scores: List[FunctionScore], baseline: Dict[str, float]) -> int:
    current = {s.key: s.crap for s in scores}
    new_offenders = [k for k, v in current.items() if v > THRESHOLD and k not in baseline]
    if new_offenders:
        print("ERROR: new functions above the threshold cannot be grandfathered via --update:")
        for key in sorted(new_offenders):
            print(f"  {key}: CRAP {current[key]:.1f}")
        return 1
    updated = {k: min(ceiling, current[k]) for k, ceiling in baseline.items() if current.get(k, 0.0) > THRESHOLD}
    _write_baseline(updated)
    print(f"Ratcheted CRAP baseline: {len(updated)} grandfathered function(s) (was {len(baseline)}).")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--coverage-json", default="coverage.json", help="coverage.py JSON report (default: coverage.json)"
    )
    parser.add_argument("--top", type=int, default=20, help="how many of the worst functions to print")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--init", action="store_true", help="(re)write the baseline from the current offenders")
    mode.add_argument("--update", action="store_true", help="lower ceilings / drop fixed entries (never raise)")
    mode.add_argument("--report-only", action="store_true", help="print the report and always exit 0")
    args = parser.parse_args(argv)

    cov_path = Path(args.coverage_json)
    if not cov_path.is_absolute():
        cov_path = ROOT / cov_path
    if not cov_path.exists():
        print(f"ERROR: {cov_path} not found. Run pytest with --cov-report=json first.")
        return 2
    scores, unmeasured = score_functions(json.loads(cov_path.read_text(encoding="utf-8")))
    _print_report(scores, unmeasured, args.top)

    if args.report_only:
        return 0
    if args.init:
        _write_baseline({s.key: s.crap for s in scores if s.crap > THRESHOLD})
        print(f"Initialized {BASELINE_PATH.relative_to(ROOT)}.")
        return 0
    baseline = load_baseline()
    return run_update(scores, baseline) if args.update else run_check(scores, baseline)


if __name__ == "__main__":
    sys.exit(main())
