# Objective: Tests for the programmatic graders that decide correctness of model answers in benchmark v2.
"""Graders decide every score in the benchmark, so their failure modes matter.

Two kinds of mistake are fatal and both are covered here: accepting a wrong
answer (a model gets credit it did not earn) and rejecting a right one because
of formatting (a model loses credit for writing ``1.234,50`` instead of
``1234.5``).
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

from scripts.benchmark_v2.graders import (  # noqa: E402
    extract_final_number,
    grade,
    is_abstention,
    parse_number,
    reference_answer,
)

# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token,expected",
    [
        ("42", 42.0),
        ("-7", -7.0),
        ("3.14", 3.14),  # lone dot is decimal
        ("0.625", 0.625),  # leading zero must stay decimal
        ("1,5", 1.5),  # pt-BR decimal comma
        ("1.234,56", 1234.56),  # pt-BR thousands + decimal
        ("1,234.56", 1234.56),  # en-US thousands + decimal
        ("1.234.567", 1234567.0),  # two triples: unambiguous thousands
        ("2.5e3", 2500.0),
    ],
)
def test_parse_number_handles_both_conventions(token, expected):
    assert parse_number(token) == pytest.approx(expected)


def test_parse_number_rejects_non_numeric():
    assert parse_number("abc") is None
    assert parse_number("") is None


def test_final_number_prefers_the_text_after_the_answer_marker():
    answer = "Primeiro calculei 120 e depois 45. Resposta: 165"
    assert extract_final_number(answer) == 165.0


def test_final_number_falls_back_to_the_last_number_without_a_marker():
    assert extract_final_number("O total do primeiro passo foi 12, e do segundo 30") == 30.0


def test_final_number_is_none_when_there_is_no_number():
    assert extract_final_number("nao sei responder") is None


# ---------------------------------------------------------------------------
# Numeric grading
# ---------------------------------------------------------------------------


def numeric_item(gold, tolerance=1e-3):
    return {"grader": "numeric", "answer_type": "numeric", "gold": gold, "tolerance": tolerance}


def test_numeric_accepts_within_tolerance_and_rejects_outside():
    item = numeric_item(110.4)
    assert grade("Resposta: 110,40", item).correct is True
    assert grade("Resposta: 110.4", item).correct is True
    assert grade("Resposta: 111", item).correct is False


def test_numeric_tolerance_is_relative_to_the_magnitude():
    assert grade("Resposta: 1000.5", numeric_item(1000.0)).correct is True
    assert grade("Resposta: 1002", numeric_item(1000.0)).correct is False


def test_numeric_zero_gold_uses_the_absolute_floor():
    assert grade("Resposta: 0", numeric_item(0.0)).correct is True
    assert grade("Resposta: 1", numeric_item(0.0)).correct is False


def test_numeric_reports_a_missing_number_instead_of_crashing():
    result = grade("Nao consegui calcular.", numeric_item(5.0))
    assert result.correct is False
    assert "nenhum numero" in result.detail


def test_numeric_ignores_the_reasoning_and_reads_the_final_value():
    item = numeric_item(29.09, tolerance=1e-3)
    answer = "Convertendo 8 GB para 8000 MB e dividindo por 275 MB/s, obtemos. Resposta: 29,09"
    assert grade(answer, item).correct is True


# ---------------------------------------------------------------------------
# Exact, MCQ, set
# ---------------------------------------------------------------------------


def test_exact_accepts_any_listed_spelling_and_ignores_accents():
    item = {"grader": "exact", "answer_type": "exact", "gold": ["3FN", "terceira forma normal"]}
    assert grade("Resposta: 3FN", item).correct is True
    assert grade("A relacao esta na Terceira Forma Normal.", item).correct is True
    assert grade("Resposta: BCNF", item).correct is False


def test_exact_accepts_a_scalar_gold():
    item = {"grader": "exact", "answer_type": "exact", "gold": "TRUNCATE"}
    assert grade("Use o comando TRUNCATE TABLE.", item).correct is True


def test_mcq_reads_the_last_letter_after_the_marker():
    item = {"grader": "mcq", "answer_type": "mcq", "gold": "C"}
    assert grade("Analisando A e B... Resposta: C", item).correct is True
    assert grade("Resposta: B", item).correct is False


def test_set_comparison_ignores_order_and_notation():
    item = {"grader": "set", "answer_type": "set", "gold": ["A", "B", "D"]}
    assert grade("Resposta: {D, A, B}", item).correct is True
    assert grade("Resposta: ABD", item).correct is True
    assert grade("Resposta: {A, B}", item).correct is False


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


def sql_item(gold):
    return {"grader": "sql", "answer_type": "sql", "gold": gold}


def test_sql_accepts_a_different_formulation_with_the_same_result():
    item = sql_item("SELECT nome FROM aluno WHERE orientador_id IS NULL")
    answer = "```sql\nSELECT a.nome FROM aluno a WHERE a.orientador_id IS NULL;\n```"
    assert grade(answer, item).correct is True


def test_sql_rejects_the_inner_join_trap():
    # The gold keeps students with no enrollment; an INNER JOIN silently drops them.
    item = sql_item(
        "SELECT a.nome FROM aluno a LEFT JOIN matricula m ON m.aluno_id = a.id WHERE m.aluno_id IS NULL"
    )
    answer = "SELECT a.nome FROM aluno a JOIN matricula m ON m.aluno_id = a.id WHERE m.aluno_id IS NULL"
    assert grade(answer, item).correct is False


def test_sql_reports_an_invalid_query_as_wrong_not_as_an_error():
    result = grade("SELECT nome FROM inexistente", sql_item("SELECT nome FROM aluno"))
    assert result.correct is False
    assert "invalida" in result.detail


def test_sql_without_a_select_is_wrong():
    assert grade("Nao sei escrever essa consulta.", sql_item("SELECT 1")).correct is False


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------


def test_abstention_is_correct_only_when_expected():
    expected = {"grader": "abstention", "expected_abstention": True}
    assert grade("Nao consta no documento fornecido.", expected).correct is True
    assert grade("O prazo e de 7 dias.", expected).correct is False


def test_invented_answer_where_abstention_was_required_is_wrong():
    item = {"grader": "abstention", "expected_abstention": True}
    assert grade("Segundo o regulamento, o prazo e de 3 dias uteis.", item).correct is False


@pytest.mark.parametrize(
    "text",
    ["Nao sei.", "Informacao insuficiente para responder.", "I don't know", "nao ha informacao no texto"],
)
def test_abstention_patterns(text):
    assert is_abstention(text)


def test_a_normal_answer_is_not_an_abstention():
    assert not is_abstention("A resposta e 42.")


# ---------------------------------------------------------------------------
# Rubric and reference answers
# ---------------------------------------------------------------------------


def test_rubric_items_defer_to_a_judge():
    result = grade("qualquer texto", {"grader": "rubric", "answer_type": "rubric"})
    assert result.correct is None
    assert result.needs_judge


@pytest.mark.parametrize(
    "item",
    [
        {"grader": "numeric", "answer_type": "numeric", "gold": 6.5, "tolerance": 1e-6},
        {"grader": "exact", "answer_type": "exact", "gold": ["ITV-AV-08"]},
        {"grader": "set", "answer_type": "set", "gold": ["A", "C"]},
        {"grader": "sql", "answer_type": "sql", "gold": "SELECT COUNT(*) FROM aluno"},
        {"grader": "abstention", "expected_abstention": True},
    ],
)
def test_reference_answer_is_accepted_by_its_own_grader(item):
    assert grade(reference_answer(item), item).correct is True


def test_unknown_grader_is_a_loud_failure():
    with pytest.raises(KeyError):
        grade("x", {"grader": "inexistente"})
