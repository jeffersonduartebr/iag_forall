# Objective: Tests for the SLOC checker's format-invariant logical counting.
"""Lock in that `check_file_length` counts logical lines invariant to ruff wrapping."""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_file_length.py"
_spec = importlib.util.spec_from_file_location("check_file_length", _SCRIPT)
cfl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfl)


def test_logical_sloc_ignores_blank_and_comment_lines():
    assert cfl._logical_sloc("x = 1\n\n# comment\ny = 2\n") == 2


def test_logical_sloc_invariant_to_line_wrapping():
    one_line = "result = foo(aaaaa, bbbbb, ccccc, ddddd)\n"
    wrapped = "result = foo(\n    aaaaa,\n    bbbbb,\n    ccccc,\n    ddddd,\n)\n"
    # This is the whole point: how ruff wraps a statement must not change the count.
    assert cfl._logical_sloc(one_line) == cfl._logical_sloc(wrapped) == 1


def test_logical_sloc_counts_multiline_string_as_one_statement():
    assert cfl._logical_sloc('doc = """\nline1\nline2\nline3\n"""\n') == 1


def test_logical_sloc_falls_back_when_untokenizable():
    # Unterminated string breaks tokenize → physical non-blank/non-comment fallback.
    assert cfl._logical_sloc('x = "unterminated\n') == 1


def test_run_check_flags_over_baseline():
    counts = {"a.py": 501, "b.py": 400}
    assert cfl.run_check(counts, {}) == 1  # a.py over MAX_SLOC, no baseline
    assert cfl.run_check({"a.py": 501}, {"a.py": 501}) == 0  # within its ceiling


def test_run_update_refuses_new_over_limit_file():
    # A file over the limit that is not already baselined cannot be grandfathered.
    assert cfl.run_update({"new.py": 600}, {}) == 1
