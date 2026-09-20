# Objective: Programmatic verifiers that decide whether a model answer to a benchmark v2 item is correct.
"""Graders: the difference between a benchmark and a vibe check.

Every grader takes the model's raw answer text plus the item and returns a
:class:`GradeResult`. ``correct=None`` means "no program can decide this" and the
item must go to a judge — only ``rubric`` items are allowed to land there.

The parsers are deliberately generous about *form* (markdown, prose around the
number, ``Resposta:`` markers, thousands separators) and strict about *value*.
A model that gets the arithmetic right must not lose a point for formatting.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class GradeResult:
    """Outcome of grading one answer."""

    correct: Optional[bool]
    extracted: Any = None
    detail: str = ""

    @property
    def needs_judge(self) -> bool:
        return self.correct is None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

#: Markers a model uses to flag its final answer; the text after the last one wins.
ANSWER_MARKERS = (
    "resposta final",
    "resposta:",
    "resposta e",
    "portanto",
    "logo,",
    "final answer",
    "answer:",
)

ABSTENTION_PATTERNS = (
    "nao sei",
    "nao e possivel determinar",
    "nao tenho informacao",
    "nao consta",
    "nao ha informacao",
    "informacao insuficiente",
    "dados insuficientes",
    "nao posso responder",
    "nao foi possivel encontrar",
    # normalize() strips apostrophes, so these are written the way they arrive there.
    "i dont know",
    "cannot determine",
    "insufficient information",
)


def strip_accents(text: str) -> str:
    """Accent-free copy of ``text`` (comparisons ignore diacritics)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str) -> str:
    """Lowercase, accent-free, punctuation-free, whitespace-collapsed form."""
    cleaned = strip_accents(str(text)).lower()
    cleaned = cleaned.replace("'", "").replace("’", "")  # don't -> dont, not "don t"
    cleaned = re.sub(r"[^\w\s./-]", " ", cleaned)
    return " ".join(cleaned.split())


def tail_after_marker(text: str) -> str:
    """Text after the last answer marker, or the whole text when there is none."""
    flat = strip_accents(text).lower()
    best = -1
    for marker in ANSWER_MARKERS:
        found = flat.rfind(marker)
        if found > best:
            best = found + len(marker)
    return text[best:] if best >= 0 else text


def is_abstention(text: str) -> bool:
    """True when the answer declines to answer instead of inventing one."""
    flat = normalize(text)
    return any(pattern in flat for pattern in ABSTENTION_PATTERNS)


# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"[-+]?\d[\d.,]*(?:\s*[eE]\s*[-+]?\d+)?")


def parse_number(token: str) -> Optional[float]:
    """Read one numeric token written in either pt-BR or en-US convention.

    Both separators present means the last one is the decimal mark. A lone comma
    is always decimal. A lone dot is read as a decimal point unless the token has
    two or more dot-separated triples (``1.234.567``), the only shape where the
    thousands reading is unambiguous. ``1.234`` stays 1.234: the statements ask
    for rounded decimals, so that is what a dot almost always means here.
    """
    token = token.strip().replace(" ", "")
    if not token:
        return None
    if "," in token and "." in token:
        decimal_sep = "," if token.rfind(",") > token.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        token = token.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif "," in token:
        token = token.replace(",", ".")
    elif token.count(".") >= 2:
        head, *rest = token.split(".")
        if all(len(part) == 3 for part in rest) and head.lstrip("-+").isdigit():
            token = head + "".join(rest)
    try:
        return float(token)
    except ValueError:
        return None


def extract_numbers(text: str) -> List[float]:
    """Every number in ``text``, in order of appearance."""
    values = [parse_number(match.group()) for match in _NUMBER_RE.finditer(text)]
    return [v for v in values if v is not None]


def extract_final_number(text: str) -> Optional[float]:
    """The model's final numeric answer: last number after the answer marker."""
    for candidate in (tail_after_marker(text), text):
        numbers = extract_numbers(candidate)
        if numbers:
            return numbers[-1]
    return None


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------


def grade_numeric(answer: str, item: Dict[str, Any]) -> GradeResult:
    """Compare the final number against ``gold`` within ``tolerance`` (relative)."""
    gold = float(item["gold"])
    tolerance = float(item.get("tolerance") or 0.0)
    value = extract_final_number(answer)
    if value is None:
        return GradeResult(False, None, "nenhum numero encontrado na resposta")
    limit = max(abs(gold) * tolerance, tolerance)
    ok = abs(value - gold) <= limit
    return GradeResult(ok, value, f"esperado {gold} +-{limit:g}, obtido {value}")


def grade_exact(answer: str, item: Dict[str, Any]) -> GradeResult:
    """Accept when any accepted spelling of ``gold`` appears in the final answer."""
    accepted = item["gold"] if isinstance(item["gold"], list) else [item["gold"]]
    tail = normalize(tail_after_marker(answer))
    whole = normalize(answer)
    for option in accepted:
        needle = normalize(option)
        if needle and (needle in tail or needle in whole):
            return GradeResult(True, option, f"encontrou '{option}'")
    return GradeResult(False, None, f"nenhuma das formas aceitas: {accepted}")


def grade_mcq(answer: str, item: Dict[str, Any]) -> GradeResult:
    """Extract the chosen option letter and compare it with ``gold``."""
    gold = str(item["gold"]).strip().upper()
    tail = tail_after_marker(answer)
    matches = re.findall(r"\b([A-E])\b[).:\s]?", strip_accents(tail).upper())
    if not matches:
        matches = re.findall(r"\b([A-E])\b[).:\s]", strip_accents(answer).upper())
    if not matches:
        return GradeResult(False, None, "nenhuma alternativa identificada")
    chosen = matches[-1]
    return GradeResult(chosen == gold, chosen, f"esperado {gold}, obtido {chosen}")


def _parse_set(text: str) -> frozenset:
    """Read ``{A, B}`` / ``A,B`` / ``AB`` into a set of attribute names."""
    inner = re.findall(r"\{([^}]*)\}", text)
    raw = inner[-1] if inner else text
    parts = [p.strip() for p in re.split(r"[,;+\s]+", raw) if p.strip()]
    if len(parts) == 1 and len(parts[0]) > 1 and parts[0].isalpha() and parts[0].isupper():
        parts = list(parts[0])  # "AB" means {A, B}
    return frozenset(p.upper() for p in parts)


def grade_set(answer: str, item: Dict[str, Any]) -> GradeResult:
    """Compare an unordered set of attributes (closure, candidate key)."""
    gold = _parse_set(",".join(item["gold"]) if isinstance(item["gold"], list) else str(item["gold"]))
    tail = tail_after_marker(answer)
    found = _parse_set(tail)
    if not found:
        return GradeResult(False, None, "nenhum conjunto identificado")
    return GradeResult(found == gold, sorted(found), f"esperado {sorted(gold)}, obtido {sorted(found)}")


def grade_abstention(answer: str, item: Dict[str, Any]) -> GradeResult:
    """Correct when the model abstains exactly where it was supposed to."""
    abstained = is_abstention(answer)
    expected = bool(item.get("expected_abstention", True))
    return GradeResult(abstained == expected, abstained, f"esperava abstencao={expected}, obteve {abstained}")


def grade_rubric(answer: str, item: Dict[str, Any]) -> GradeResult:
    """Dissertative item: no program decides it, the judge does."""
    return GradeResult(None, None, "requer juiz com rubrica")


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


def _extract_sql(text: str) -> Optional[str]:
    """Pull the SQL statement out of a fenced block or raw prose."""
    fenced = re.findall(r"```(?:sql)?\s*(.+?)```", text, flags=re.S | re.I)
    candidate = fenced[-1] if fenced else text
    match = re.search(r"\b(SELECT|WITH)\b.+", candidate, flags=re.S | re.I)
    if not match:
        return None
    statement = match.group().strip()
    statement = statement.split(";")[0]
    return statement.strip()


def _rows_equal(expected: Sequence[tuple], actual: Sequence[tuple], ordered: bool) -> bool:
    """Compare result sets, ignoring row order unless the gold query orders them."""
    if ordered:
        return list(expected) == list(actual)
    return sorted(map(repr, expected)) == sorted(map(repr, actual))


def grade_sql(answer: str, item: Dict[str, Any]) -> GradeResult:
    """Run the candidate query against the fixture and compare it with the gold query.

    Correctness is the result set, not the text: any query returning the same
    rows counts, which is the only fair way to grade SQL.
    """
    from .sql_fixture import connect

    statement = _extract_sql(answer)
    if not statement:
        return GradeResult(False, None, "nenhuma consulta SELECT encontrada")
    with connect() as conn:
        try:
            expected = conn.execute(item["gold"]).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - gold is tested at build time
            return GradeResult(None, None, f"consulta de gabarito invalida: {exc}")
        try:
            actual = conn.execute(statement).fetchall()
        except sqlite3.Error as exc:
            return GradeResult(False, statement, f"consulta invalida: {exc}")
    ordered = "order by" in item["gold"].lower()
    ok = _rows_equal(expected, actual, ordered)
    return GradeResult(ok, statement, f"{len(expected)} linhas esperadas, {len(actual)} obtidas")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRADERS: Dict[str, Callable[[str, Dict[str, Any]], GradeResult]] = {
    "numeric": grade_numeric,
    "exact": grade_exact,
    "mcq": grade_mcq,
    "set": grade_set,
    "sql": grade_sql,
    "abstention": grade_abstention,
    "rubric": grade_rubric,
}


def grade(answer: str, item: Dict[str, Any]) -> GradeResult:
    """Grade one answer with the grader named by the item."""
    name = item.get("grader") or item.get("answer_type") or "rubric"
    grader = GRADERS.get(name)
    if grader is None:
        raise KeyError(f"grader desconhecido: {name}")
    return grader(answer, item)


def reference_answer(item: Dict[str, Any]) -> str:
    """A minimal answer that the item's own grader must accept.

    Used by the build step to self-test every generated item: a gold value whose
    grader rejects it means the item is broken and must not enter the corpus.
    """
    if item.get("expected_abstention"):
        return "Nao e possivel determinar com a informacao disponivel."
    gold = item.get("gold")
    if item["grader"] == "set":
        values = gold if isinstance(gold, list) else [gold]
        return "Resposta: {" + ", ".join(str(v) for v in values) + "}"
    if item["grader"] == "sql":
        return f"```sql\n{gold}\n```"
    if item["grader"] == "exact" and isinstance(gold, list):
        gold = gold[0]
    return f"Resposta: {gold}"
