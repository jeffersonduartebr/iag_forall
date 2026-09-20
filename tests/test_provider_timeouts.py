# Objective: The timeout the router computes must reach the provider SDKs.
"""``timeout_seconds`` was computed per workload class and then dropped.

Only the Ollama adapter read it. The OpenAI, Anthropic and Gemini paths passed
nothing to their SDKs and fell back to the ten-minute defaults. The global
deadline still cancelled the ``await``, so no client hung — but the connection
and the token generation continued on the provider's side, and were billed.

Gemini was the worst case: its SDK is synchronous, so the call runs in a thread,
and a thread is not cancellable. Sixteen slow Gemini calls used to drain the
default executor and stall every ``asyncio.to_thread`` in the application.
"""

import inspect

import pytest
from app.providers._timeouts import CLOUD_DEFAULT_TIMEOUT_S, resolve_timeout


def test_the_routers_timeout_is_honoured():
    assert resolve_timeout({"timeout_seconds": 35}) == 35.0


def test_a_missing_timeout_falls_back_to_the_ceiling_not_to_ten_minutes():
    assert resolve_timeout({}) == CLOUD_DEFAULT_TIMEOUT_S
    assert CLOUD_DEFAULT_TIMEOUT_S < 600, "o default do SDK é 600 s; o tecto tem de ser menor"


def test_a_nonsensical_timeout_is_clamped():
    """A zero or negative deadline would mean 'fail immediately', which is not
    what an exhausted budget should do to a request already in flight."""
    assert resolve_timeout({"timeout_seconds": 0}) == 1.0
    assert resolve_timeout({"timeout_seconds": -5}) == 1.0


@pytest.mark.parametrize("module_name", ["_openai", "_anthropic"])
def test_the_cloud_sdks_receive_a_timeout(module_name):
    from app.providers import _anthropic, _openai

    module = {"_openai": _openai, "_anthropic": _anthropic}[module_name]
    source = inspect.getsource(module)
    assert "resolve_timeout(kwargs)" in source, module_name


def test_gemini_runs_off_the_shared_executor():
    """Its thread is not cancellable, so it must not queue with everything else."""
    from app.providers import _gemini

    source = inspect.getsource(_gemini)
    assert "run_blocking_provider(_call)" in source
    assert "asyncio.to_thread(_call)" not in source


def test_the_blocking_pool_is_bounded_and_separate():
    from app.utils.executors import get_blocking_provider_executor, get_cpu_executor

    blocking = get_blocking_provider_executor()
    assert blocking is not get_cpu_executor()
    assert blocking._max_workers >= 2
