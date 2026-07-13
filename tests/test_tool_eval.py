# Objective: Test coverage for tool-calling (function-calling) eval scoring.
"""Unit tests for the offline tool-calling evaluation harness (BFCL-style)."""

from typing import Any, Dict, List, Optional, Tuple

import pytest
from app.services import tool_eval


def _tool_call(name: str, arguments: Any) -> Dict[str, Any]:
    """Build a normalized provider tool-call payload."""
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}


def _weather_case(expected_arguments: Optional[Dict[str, Any]] = None) -> tool_eval.ToolEvalCase:
    return tool_eval.ToolEvalCase(
        case_id="weather",
        prompt="Weather in Paris?",
        tools=[],
        expected_tool_name="get_weather",
        expected_arguments=expected_arguments,
    )


def test_score_correct_tool_and_matching_arguments():
    """A correct tool with matching JSON arguments scores a perfect 1.0."""
    case = _weather_case({"city": "Paris"})
    row = tool_eval.score_tool_call([_tool_call("get_weather", '{"city": "Paris"}')], case)
    assert row["called_a_tool"] is True
    assert row["correct_tool"] is True
    assert row["valid_json_arguments"] is True
    assert row["arguments_match"] is True
    assert row["score"] == pytest.approx(1.0)


def test_score_incorrect_tool():
    """Selecting the wrong tool zeroes the tool-selection weight."""
    case = _weather_case({"city": "Paris"})
    row = tool_eval.score_tool_call([_tool_call("search", '{"query": "Paris"}')], case)
    assert row["correct_tool"] is False
    assert row["valid_json_arguments"] is True
    assert row["arguments_match"] is False
    # Only the valid-JSON component (0.2) is credited.
    assert row["score"] == pytest.approx(0.2)


def test_score_invalid_json_arguments():
    """Unparseable arguments fail JSON validity and argument matching."""
    case = _weather_case({"city": "Paris"})
    row = tool_eval.score_tool_call([_tool_call("get_weather", "{not-json")], case)
    assert row["correct_tool"] is True
    assert row["valid_json_arguments"] is False
    assert row["arguments_match"] is False
    assert row["parsed_arguments"] is None
    # Correct tool only (0.6).
    assert row["score"] == pytest.approx(0.6)


def test_score_argument_mismatch():
    """Correct tool + valid JSON but wrong argument values misses the arg weight."""
    case = _weather_case({"city": "Paris"})
    row = tool_eval.score_tool_call([_tool_call("get_weather", '{"city": "London"}')], case)
    assert row["correct_tool"] is True
    assert row["valid_json_arguments"] is True
    assert row["arguments_match"] is False
    assert row["score"] == pytest.approx(0.8)


def test_score_no_tool_called():
    """No tool call means nothing is credited and the score is zero."""
    case = _weather_case({"city": "Paris"})
    row = tool_eval.score_tool_call([], case)
    assert row["called_a_tool"] is False
    assert row["correct_tool"] is False
    assert row["valid_json_arguments"] is False
    assert row["arguments_match"] is False
    assert row["score"] == pytest.approx(0.0)


def test_score_without_expected_arguments_redistributes_weight():
    """When no expected arguments are set, arg-match is None and weight renormalizes."""
    case = _weather_case(expected_arguments=None)
    row = tool_eval.score_tool_call([_tool_call("get_weather", '{"city": "Paris"}')], case)
    assert row["arguments_match"] is None
    # correct (0.6) + valid_json (0.2) renormalized over 0.8 -> 1.0.
    assert row["score"] == pytest.approx(1.0)

    wrong = tool_eval.score_tool_call([_tool_call("search", '{"q": "x"}')], case)
    assert wrong["arguments_match"] is None
    # valid_json only: 0.2 / 0.8 = 0.25.
    assert wrong["score"] == pytest.approx(0.25)


def test_score_accepts_dict_arguments():
    """Some providers deliver arguments already parsed as a dict."""
    case = _weather_case({"city": "Paris"})
    row = tool_eval.score_tool_call([_tool_call("get_weather", {"city": "Paris"})], case)
    assert row["valid_json_arguments"] is True
    assert row["arguments_match"] is True


def test_aggregate_tool_eval_means():
    """Aggregation computes accuracy, validity, match rate, mean score and n."""
    scores = [
        {"correct_tool": True, "valid_json_arguments": True, "arguments_match": True, "score": 1.0},
        {"correct_tool": True, "valid_json_arguments": True, "arguments_match": False, "score": 0.8},
        {"correct_tool": False, "valid_json_arguments": False, "arguments_match": None, "score": 0.0},
    ]
    agg = tool_eval.aggregate_tool_eval(scores)
    assert agg["n"] == 3
    assert agg["tool_selection_accuracy"] == pytest.approx(2 / 3)
    assert agg["argument_validity_rate"] == pytest.approx(2 / 3)
    # Only the two rows with non-None arguments_match count: 1 of 2.
    assert agg["argument_match_rate"] == pytest.approx(0.5)
    assert agg["mean_score"] == pytest.approx((1.0 + 0.8 + 0.0) / 3)


def test_aggregate_empty_scores():
    """Empty aggregation returns zeroed rates without dividing by zero."""
    agg = tool_eval.aggregate_tool_eval([])
    assert agg == {
        "tool_selection_accuracy": 0.0,
        "argument_validity_rate": 0.0,
        "argument_match_rate": 0.0,
        "mean_score": 0.0,
        "n": 0,
    }


def test_golden_set_is_serializable_and_nonempty():
    """The built-in golden set exposes 3-5 serializable cases."""
    assert 3 <= len(tool_eval.TOOL_EVAL_GOLDEN_SET) <= 5
    dicts = tool_eval.tool_eval_golden_set_as_dicts()
    assert all("prompt" in row and "expected_tool_name" in row for row in dicts)
    names = {row["expected_tool_name"] for row in dicts}
    assert {"get_weather", "search"}.issubset(names)


@pytest.mark.asyncio
async def test_run_tool_eval_with_mocked_call_model():
    """run_tool_eval drives cases through an injected mock call_model (no network)."""
    calls: List[Tuple[str, str]] = []

    async def fake_call_model(
        model: str,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        calls.append((model, prompt))
        # Echo back the correct tool + arguments for weather cases; wrong for others.
        if "weather" in prompt.lower() or "Tokyo" in prompt or "Paris" in prompt:
            city = "Tokyo" if "Tokyo" in prompt else "Paris"
            meta = {
                "tool_calls": [_tool_call("get_weather", '{"city": "%s"}' % city)],
                "finish_reason": "tool_calls",
            }
            return "", meta
        meta = {
            "tool_calls": [_tool_call("search", '{"query": "python asyncio tutorial"}')],
            "finish_reason": "tool_calls",
        }
        return "", meta

    result = await tool_eval.run_tool_eval(
        "test/model",
        tool_eval.TOOL_EVAL_GOLDEN_SET,
        call_model=fake_call_model,
    )

    assert result["model"] == "test/model"
    assert len(result["results"]) == len(tool_eval.TOOL_EVAL_GOLDEN_SET)
    assert len(calls) == len(tool_eval.TOOL_EVAL_GOLDEN_SET)
    agg = result["aggregate"]
    assert agg["n"] == len(tool_eval.TOOL_EVAL_GOLDEN_SET)
    # Every case selected an available (correct) tool in this mock.
    assert agg["tool_selection_accuracy"] == pytest.approx(1.0)
    assert agg["argument_validity_rate"] == pytest.approx(1.0)
    assert result["results"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_run_tool_eval_captures_provider_errors():
    """A raising call_model is captured per-row instead of aborting the run."""

    async def boom(*_args: Any, **_kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        raise RuntimeError("provider down")

    result = await tool_eval.run_tool_eval("m", [_weather_case({"city": "Paris"})], call_model=boom)
    row = result["results"][0]
    assert row["error"] == "provider down"
    assert row["called_a_tool"] is False
    assert result["aggregate"]["mean_score"] == pytest.approx(0.0)
