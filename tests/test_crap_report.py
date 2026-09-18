# Objective: Test coverage for the CRAP report/ratchet script.
"""scripts/crap_report.py: CRAP formula, coverage join and ratchet semantics."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "crap_report", Path(__file__).resolve().parents[1] / "scripts" / "crap_report.py"
)
crap = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(crap)


def _score(key, cc, cov):
    path = key.split("::")[0]
    return crap.FunctionScore(key, path, 1, cc, cov, crap.crap_score(cc, cov))


def test_crap_formula():
    assert crap.crap_score(1, 1.0) == 1
    assert crap.crap_score(10, 0.0) == 110
    assert crap.crap_score(10, 1.0) == 10
    # CC 25 com 81% de cobertura fica perto do limite de 30.
    assert crap.crap_score(25, 0.81) == pytest.approx(29.29, abs=0.01)


def test_function_index_combines_statements_and_branches():
    report = {
        "files": {
            "app/app/x.py": {
                "functions": {
                    "f": {
                        "start_line": 10,
                        "summary": {
                            "num_statements": 6,
                            "covered_lines": 3,
                            "num_branches": 4,
                            "covered_branches": 2,
                        },
                    },
                    "": {"start_line": 1, "summary": {}},  # módulo: ignorado
                }
            }
        }
    }
    assert crap._function_index(report) == {("app/app/x.py", 10): ("f", 0.5)}


def test_check_flags_new_offenders_and_growth(monkeypatch, capsys):
    baseline = {"a.py::old": 100.0}
    scores = [_score("a.py::old", 10, 0.0), _score("a.py::new", 12, 0.0), _score("a.py::ok", 3, 0.0)]
    # old: 110 > 100 (cresceu); new: 156 > 30 (nova); ok: 12 <= 30
    assert crap.run_check(scores, baseline) == 1
    out = capsys.readouterr().out
    assert "a.py::old" in out and "a.py::new" in out and "a.py::ok" not in out

    assert crap.run_check([_score("a.py::old", 10, 0.1)], baseline) == 0  # 82.9 <= 100


def test_update_only_lowers_and_refuses_new(monkeypatch, tmp_path):
    monkeypatch.setattr(crap, "BASELINE_PATH", tmp_path / "crap_baseline.json")
    baseline = {"a.py::old": 100.0, "a.py::fixed": 80.0}
    scores = [_score("a.py::old", 10, 0.1), _score("a.py::fixed", 5, 1.0)]
    assert crap.run_update(scores, baseline) == 0
    assert crap.load_baseline() == {"a.py::old": pytest.approx(82.9, abs=0.1)}

    assert crap.run_update(scores + [_score("a.py::new", 12, 0.0)], crap.load_baseline()) == 1
