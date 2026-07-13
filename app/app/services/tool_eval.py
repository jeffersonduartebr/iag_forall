# Objective: Application runtime code for tool-calling (function-calling) evaluation scoring.
"""Self-contained scoring for tool-calling / function-calling accuracy (BFCL-style).

This module measures how well a model selects the *right* tool and emits *valid*
structured arguments, independent of the free-text quality judged elsewhere. It
is intentionally free of network and provider side effects: scoring works on the
normalized ``meta["tool_calls"]`` payload produced by ``providers_async.call_model``
and the driver accepts an injectable ``call_model`` for offline unit testing.

A tool call is the normalized shape::

    {"id": ..., "type": "function",
     "function": {"name": <str>, "arguments": <json str>}}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

# Async callable that mirrors ``providers_async.call_model`` and returns the
# legacy ``(text, meta)`` tuple. Kept permissive (``...``) so both the real
# provider entry point and lightweight mocks satisfy it.
CallModelFn = Callable[..., Awaitable[Tuple[str, Dict[str, Any]]]]

# Score weights (sum to 1.0 when arguments are checked). When a case omits
# ``expected_arguments`` the argument-match weight is redistributed across the
# remaining components so a perfect answer still scores 1.0.
W_CORRECT_TOOL = 0.6
W_VALID_JSON = 0.2
W_ARGS_MATCH = 0.2


@dataclass
class ToolEvalCase:
    """One tool-calling evaluation case.

    Attributes:
        prompt: User prompt to send to the model.
        tools: OpenAI-style tool schema list passed to the provider.
        expected_tool_name: Name of the tool the model should select first.
        expected_arguments: Exact arguments expected; when ``None`` argument
            matching is skipped (only tool selection and JSON validity scored).
        tool_choice: Optional provider ``tool_choice`` directive.
        case_id: Stable identifier for reporting rows.
    """

    prompt: str
    tools: List[Dict[str, Any]]
    expected_tool_name: str
    expected_arguments: Optional[Dict[str, Any]] = None
    tool_choice: Optional[Any] = None
    case_id: str = ""


def _first_tool_call(result_tool_calls: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Return the first tool call mapping, or ``None`` when none were emitted."""
    if not result_tool_calls:
        return None
    first = result_tool_calls[0]
    if not isinstance(first, dict):
        return None
    return first


def _parse_arguments(raw: Any) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Best-effort parse of a tool call's ``arguments`` field.

    Returns ``(valid_json, parsed_dict_or_none)``. Providers usually emit a JSON
    string, but some already deliver a dict; both are accepted. Anything that is
    not a JSON object is treated as invalid for matching purposes.
    """
    if isinstance(raw, dict):
        return True, dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return False, None
        if isinstance(parsed, dict):
            return True, parsed
        # Valid JSON but not an object (e.g. a bare list/number): parseable,
        # yet not a usable argument mapping.
        return True, None
    return False, None


def score_tool_call(result_tool_calls: Optional[List[Dict[str, Any]]], case: ToolEvalCase) -> Dict[str, Any]:
    """Score a single tool-calling result against an expected case.

    Returns per-case metrics: ``called_a_tool``, ``correct_tool``,
    ``valid_json_arguments``, ``arguments_match`` (``None`` when the case has no
    ``expected_arguments``) and an overall ``score`` in ``[0, 1]``.
    """
    first = _first_tool_call(result_tool_calls)
    called_a_tool = first is not None

    called_name = ""
    raw_arguments: Any = None
    if first is not None:
        raw_function = first.get("function")
        function: Dict[str, Any] = raw_function if isinstance(raw_function, dict) else {}
        called_name = str(function.get("name") or "")
        raw_arguments = function.get("arguments")

    correct_tool = called_a_tool and called_name == case.expected_tool_name
    valid_json_arguments, parsed_arguments = _parse_arguments(raw_arguments)
    if not called_a_tool:
        valid_json_arguments = False

    arguments_match: Optional[bool]
    if case.expected_arguments is None:
        arguments_match = None
    else:
        arguments_match = parsed_arguments == case.expected_arguments

    score = _compute_score(correct_tool, valid_json_arguments, arguments_match)
    return {
        "case_id": case.case_id,
        "expected_tool_name": case.expected_tool_name,
        "called_tool_name": called_name,
        "called_a_tool": called_a_tool,
        "correct_tool": correct_tool,
        "valid_json_arguments": valid_json_arguments,
        "arguments_match": arguments_match,
        "parsed_arguments": parsed_arguments,
        "score": score,
    }


def _compute_score(correct_tool: bool, valid_json: bool, arguments_match: Optional[bool]) -> float:
    """Blend component signals into a normalized ``[0, 1]`` score."""
    correct_part = W_CORRECT_TOOL * (1.0 if correct_tool else 0.0)
    json_part = W_VALID_JSON * (1.0 if valid_json else 0.0)
    if arguments_match is None:
        applicable = W_CORRECT_TOOL + W_VALID_JSON
        if applicable <= 0.0:
            return 0.0
        return round((correct_part + json_part) / applicable, 6)
    args_part = W_ARGS_MATCH * (1.0 if arguments_match else 0.0)
    return round(correct_part + json_part + args_part, 6)


def aggregate_tool_eval(list_of_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-case scores into summary rates.

    ``argument_match_rate`` averages only over cases where ``arguments_match`` is
    not ``None`` (i.e. cases that declared expected arguments). All rates default
    to ``0.0`` when there is nothing to average.
    """
    n = len(list_of_scores)
    if n == 0:
        return {
            "tool_selection_accuracy": 0.0,
            "argument_validity_rate": 0.0,
            "argument_match_rate": 0.0,
            "mean_score": 0.0,
            "n": 0,
        }

    correct = sum(1 for row in list_of_scores if row.get("correct_tool"))
    valid_json = sum(1 for row in list_of_scores if row.get("valid_json_arguments"))
    scored = [float(row.get("score") or 0.0) for row in list_of_scores]

    match_rows = [row for row in list_of_scores if row.get("arguments_match") is not None]
    matched = sum(1 for row in match_rows if row.get("arguments_match"))
    argument_match_rate = (matched / len(match_rows)) if match_rows else 0.0

    return {
        "tool_selection_accuracy": correct / n,
        "argument_validity_rate": valid_json / n,
        "argument_match_rate": argument_match_rate,
        "mean_score": sum(scored) / n,
        "n": n,
    }


def _weather_tool() -> Dict[str, Any]:
    """OpenAI-style schema for a ``get_weather`` tool."""
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name."}},
                "required": ["city"],
            },
        },
    }


def _search_tool() -> Dict[str, Any]:
    """OpenAI-style schema for a ``search`` tool."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for a query string.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query."}},
                "required": ["query"],
            },
        },
    }


def _force_choice(name: str) -> Dict[str, Any]:
    """Build an OpenAI ``tool_choice`` directive that forces one function."""
    return {"type": "function", "function": {"name": name}}


TOOL_EVAL_GOLDEN_SET: List[ToolEvalCase] = [
    ToolEvalCase(
        case_id="weather_paris",
        prompt="What is the weather like in Paris right now?",
        tools=[_weather_tool(), _search_tool()],
        expected_tool_name="get_weather",
        expected_arguments={"city": "Paris"},
    ),
    ToolEvalCase(
        case_id="weather_tokyo_forced",
        prompt="Tell me the current weather in Tokyo.",
        tools=[_weather_tool(), _search_tool()],
        expected_tool_name="get_weather",
        expected_arguments={"city": "Tokyo"},
        tool_choice=_force_choice("get_weather"),
    ),
    ToolEvalCase(
        case_id="search_asyncio",
        prompt="Search the web for a python asyncio tutorial.",
        tools=[_weather_tool(), _search_tool()],
        expected_tool_name="search",
        expected_arguments={"query": "python asyncio tutorial"},
    ),
    ToolEvalCase(
        case_id="search_any_args",
        prompt="Find recent news about renewable energy.",
        tools=[_weather_tool(), _search_tool()],
        expected_tool_name="search",
        expected_arguments=None,
    ),
]


def tool_eval_golden_set_as_dicts() -> List[Dict[str, Any]]:
    """Return the built-in golden set as JSON-serializable dictionaries."""
    return [asdict(case) for case in TOOL_EVAL_GOLDEN_SET]


async def _default_call_model(
    model: str,
    prompt: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Lazily delegate to the real provider entry point (kept import-light)."""
    from ..providers_async import call_model as provider_call_model

    return await provider_call_model(model, prompt, tools=tools, tool_choice=tool_choice)


async def run_tool_eval(
    model: str,
    cases: List[ToolEvalCase],
    call_model: Optional[CallModelFn] = None,
) -> Dict[str, Any]:
    """Run the tool-eval benchmark for one model over a list of cases.

    ``call_model`` defaults to the real provider entry point but can be injected
    with a mock for offline unit testing. Each case is scored independently; a
    provider error is captured on the row rather than aborting the whole run.
    """
    caller: CallModelFn = call_model if call_model is not None else _default_call_model
    rows: List[Dict[str, Any]] = []
    for case in cases:
        try:
            _text, meta = await caller(model, case.prompt, tools=case.tools, tool_choice=case.tool_choice)
            tool_calls = meta.get("tool_calls") if isinstance(meta, dict) else None
            row = score_tool_call(tool_calls, case)
            row["finish_reason"] = meta.get("finish_reason") if isinstance(meta, dict) else None
        except Exception as exc:  # pragma: no cover - defensive, exercised via mock
            row = score_tool_call(None, case)
            row["error"] = str(exc)
        rows.append(row)

    return {
        "model": model,
        "aggregate": aggregate_tool_eval(rows),
        "results": rows,
    }
