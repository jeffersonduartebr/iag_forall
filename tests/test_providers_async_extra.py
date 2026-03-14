# Objective: Test coverage for providers async extra behavior and regressions.
"""Test coverage for providers async extra behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


import asyncio
from types import SimpleNamespace

import httpx
import pybreaker
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app import providers_async as pa


def test_timeout_quality_tokens_and_factory(monkeypatch):
    """Testa timeout quality tokens and factory."""
    monkeypatch.setattr(
        pa,
        "_runtime_provider_settings",
        lambda: {
            "adaptive_timeout_enabled": True,
            "base_timeout": 10,
            "max_timeout": 1000,
            "timeout_multiplier": 2.0,
            "reasoning_multiplier": 3.0,
        },
    )

    assert pa._get_adaptive_timeout("deepseek-r1") == 30
    assert pa._get_adaptive_timeout("llama-70b") == 20
    assert pa._get_adaptive_timeout("model-x") == 15
    assert pa._get_adaptive_timeout("model-x", workload_class="simple_text") == 20
    assert pa._get_adaptive_timeout("model-x", workload_class="knowledge_lookup") == 35

    monkeypatch.setattr(
        pa,
        "_runtime_provider_settings",
        lambda: {
            "adaptive_timeout_enabled": False,
            "base_timeout": 10,
            "max_timeout": 1000,
            "timeout_multiplier": 2.0,
            "reasoning_multiplier": 3.0,
        },
    )
    assert pa._get_adaptive_timeout("any") == 1000

    assert pa.heuristic_quality_estimate("") == 0.0
    assert pa.heuristic_quality_estimate("texto. " * 50) > 4.0
    assert pa._estimate_tokens("") == 0
    assert pa._estimate_tokens("abcd") >= 1

    monkeypatch.setattr(pa, "OpenAIProvider", lambda: "openai-provider")
    monkeypatch.setattr(pa, "AnthropicProvider", lambda: "anthropic-provider")
    monkeypatch.setattr(pa, "GeminiProvider", lambda: "gemini-provider")
    monkeypatch.setattr(pa, "OllamaProvider", lambda: "ollama-provider")
    monkeypatch.setattr(pa, "is_provider_configured", lambda prefix: prefix != "xpto")
    monkeypatch.setattr(pa.ProviderFactory, "_instances", {})
    assert pa.ProviderFactory.get_provider("openai/gpt-4o") == "openai-provider"
    assert pa.ProviderFactory.get_provider("ollama/phi4:latest") == "ollama-provider"

    with pytest.raises(RuntimeError):
        pa.ProviderFactory.get_provider("xpto/model")

    monkeypatch.setattr(pa, "is_provider_configured", lambda prefix: prefix == "ollama")
    with pytest.raises(RuntimeError):
        pa.ProviderFactory.get_provider("openai/gpt-4o")


@pytest.mark.asyncio
async def test_call_model_success_and_error_categories(monkeypatch):
    """Testa call model success and error categories."""
    pa.reset_provider_runtime_state()

    class _P:
        """Represent `_P` within this module.

The class groups the state and behavior required for P."""
        async def generate(self, **kwargs):
            """Execute the generate routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return pa.LLMResponse(
                text="ok",
                latency=0.1,
                load_time=0.0,
                cost=0.01,
                prompt_tokens=10,
                completion_tokens=20,
                model_used="x",
                raw_payload="{}",
                reasoning=None,
            )

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", lambda m: _P())
    txt, meta = await pa.call_model("openai/gpt-4o", "hello")
    assert txt == "ok"
    assert meta["prompt_tokens"] == 10

    class _PCircuit:
        """Represent `_PCircuit` within this module.

The class groups the state and behavior required for PCircuit."""
        async def generate(self, **kwargs):
            """Execute the generate routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            raise pybreaker.CircuitBreakerError("open")

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", lambda m: _PCircuit())
    with pytest.raises(pa.ProviderCircuitOpenError):
        await pa.call_model("openai/gpt-4o", "x")

    class _PTimeout:
        """Represent `_PTimeout` within this module.

The class groups the state and behavior required for PTimeout."""
        async def generate(self, **kwargs):
            """Execute the generate routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            raise asyncio.TimeoutError("timeout")

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", lambda m: _PTimeout())
    with pytest.raises(pa.ProviderCallError) as exc1:
        await pa.call_model("openai/gpt-4o", "x")
    assert exc1.value.category == "provider_timeout"

    class RateLimitError(Exception):
        """Represent `RateLimitError` within this module.

The class groups the state and behavior required for RateLimitError."""
        pass

    class _PRL:
        """Represent `_PRL` within this module.

The class groups the state and behavior required for PRL."""
        async def generate(self, **kwargs):
            """Execute the generate routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            raise RateLimitError("rl")

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", lambda m: _PRL())
    with pytest.raises(pa.ProviderCallError) as exc2:
        await pa.call_model("openai/gpt-4o", "x")
    assert exc2.value.category == "provider_rate_limit"


def test_runtime_provider_settings_parses_string_and_csv_forms(monkeypatch):
    """Runtime provider settings should normalize JSON/csv warm-model inputs and aliases."""
    fake_settings = SimpleNamespace(
        get=lambda key, default=None: {
            "OLLAMA_CONCURRENCY_LIMIT": "7",
            "OLLAMA_DYNAMIC_CONCURRENCY_ENABLED": "1",
            "OLLAMA_WARM_MODELS": "qwen3.5:4b, ollama/phi4:latest",
            "OLLAMA_INTERACTIVE_MODEL_SET": '["ministral-3:3b"]',
            "MODEL_UNAVAILABLE_NEGATIVE_CACHE_TTL_SECONDS": "45",
        }.get(key, default)
    )
    monkeypatch.setitem(__import__("sys").modules, "app.settings_dynamic", SimpleNamespace(settings=fake_settings))

    cfg = pa._runtime_provider_settings()
    assert cfg["ollama_concurrency_limit"] == 7
    assert cfg["ollama_dynamic_concurrency_enabled"] is True
    assert "ollama/qwen3.5:4b" in cfg["ollama_warm_models"]
    assert cfg["ollama_interactive_warm_models"] == ["ollama/ministral-3:3b"]
    assert cfg["provider_unavailable_negative_cache_ttl_seconds"] == 45


def test_provider_runtime_state_helpers_and_expiry(monkeypatch):
    """Runtime state helpers should normalize names and expire unavailable entries."""
    pa.reset_provider_runtime_state()
    assert pa._normalize_ollama_model_name("") == "ollama/unknown"
    assert pa._normalize_ollama_model_name("phi4:latest") == "ollama/phi4:latest"

    pa._mark_ollama_model_state("phi4:latest", loaded=True, load_seconds=1.5, inflight_delta=2, queue_wait_seconds=0.4)
    entry = pa._ensure_ollama_runtime_entry("ollama/phi4:latest")
    assert entry["loaded"] is True
    assert entry["inflight"] == 2
    assert entry["last_queue_wait_seconds"] == 0.4

    pa._provider_unavailable_until["ollama/phi4:latest"] = 1.0
    monkeypatch.setattr(pa.time, "time", lambda: 10.0)
    assert pa.is_provider_temporarily_unavailable("ollama/phi4:latest") is False


@pytest.mark.asyncio
async def test_concurrency_controller_static_start_and_failed_refresh(monkeypatch):
    """Controller should publish fallback state in static mode and survive telemetry failure."""
    controller = pa.OllamaConcurrencyController()
    metric = MagicMock()
    monkeypatch.setattr(pa, "OLLAMA_DYNAMIC_CONCURRENCY_LIMIT", metric)
    monkeypatch.setattr(pa, "OLLAMA_DYNAMIC_CONCURRENCY_MODE", metric)
    monkeypatch.setattr(
        pa,
        "_runtime_provider_settings",
        lambda: {
            "ollama_concurrency_limit": 4,
            "ollama_dynamic_concurrency_enabled": False,
            "ollama_concurrency_min": 1,
            "ollama_concurrency_max": 4,
        },
    )
    await controller.start()
    assert controller.get_effective_limit() == 4

    monkeypatch.setattr(
        pa,
        "_runtime_provider_settings",
        lambda: {
            "ollama_concurrency_limit": 4,
            "ollama_dynamic_concurrency_enabled": True,
            "ollama_concurrency_min": 1,
            "ollama_concurrency_max": 4,
            "ollama_gpu_index": 0,
            "ollama_vram_high_watermark": 0.8,
            "ollama_vram_low_watermark": 0.5,
            "ollama_concurrency_step_up": 1,
            "ollama_concurrency_step_down": 1,
            "ollama_concurrency_stable_windows": 2,
        },
    )
    monkeypatch.setattr(controller, "_read_vram_snapshot", lambda gpu_index: None)
    await controller.force_refresh()
    assert controller.get_effective_limit() == 4


def test_controller_snapshot_reader_and_interactive_model_fallback(monkeypatch):
    """Helper methods should parse nvidia-smi output and fall back to global warm models."""
    controller = pa.OllamaConcurrencyController()
    monkeypatch.setattr(
        pa.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="123, 456\n"),
    )
    used, total = controller._read_vram_snapshot(0)
    assert used == 123 * 1024 * 1024
    assert total == 456 * 1024 * 1024

    monkeypatch.setattr(pa.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert controller._read_vram_snapshot(0) is None

    monkeypatch.setattr(pa, "_runtime_provider_settings", lambda: {"ollama_interactive_warm_models": [], "ollama_warm_models": ["ollama/phi4:latest"]})
    assert pa.get_interactive_ollama_models() == ["ollama/phi4:latest"]


@pytest.mark.asyncio
async def test_warm_ollama_model_runtime_success_and_failure(monkeypatch):
    """Warmup should update runtime state on success and return False on failure."""
    metric = MagicMock()
    metric.labels.return_value = metric
    monkeypatch.setattr(pa, "OLLAMA_MODEL_LOAD_SECONDS", metric)
    monkeypatch.setattr(pa, "OLLAMA_MODEL_LOADED", metric)

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"load_duration": 1_000_000_000}
    client = AsyncMock()
    client.post.return_value = response
    monkeypatch.setattr(pa, "get_http_client", AsyncMock(return_value=client))
    monkeypatch.setattr(pa, "_get_adaptive_timeout", lambda model: 30.0)
    pa.reset_provider_runtime_state()
    assert await pa.warm_ollama_model_runtime("qwen3.5:4b") is True
    assert pa._ensure_ollama_runtime_entry("ollama/qwen3.5:4b")["loaded"] is True

    client.post.side_effect = RuntimeError("down")
    assert await pa.warm_ollama_model_runtime("qwen3.5:4b") is False


def test_base_provider_metrics_helpers():
    """BaseProvider metric helpers should safely handle zero-token and failure paths."""
    class _Metric:
        def __init__(self):
            self.calls = []
        def labels(self, **kwargs):
            self.calls.append(("labels", kwargs))
            return self
        def inc(self, value=1):
            self.calls.append(("inc", value))
        def observe(self, value):
            self.calls.append(("observe", value))

    req = _Metric()
    ok = _Metric()
    err = _Metric()
    lat = _Metric()
    cost = _Metric()
    tps = _Metric()
    with patch.object(pa, "PROV_REQ", req), patch.object(pa, "PROV_OK", ok), patch.object(pa, "PROV_ERR", err), patch.object(pa, "PROV_LAT", lat), patch.object(pa, "PROV_COST", cost), patch.object(pa, "GENERATION_TOKENS_PER_SECOND", tps):
        class _Provider(pa.BaseProvider):
            async def generate(self, prompt, image_b64=None, **kwargs):
                return pa.LLMResponse(text="x", latency=0.1, cost=0.0, prompt_tokens=1, completion_tokens=1, model_used="m")
        provider = _Provider("test", 1)
        provider._record_metrics("m", 0.2, 0.3, True)
        provider._record_metrics("m", 0.2, 0.0, False)
        provider._record_generation_metrics("m", 0, 1.0)
        provider._record_generation_metrics("m", 4, 2.0)
    assert any(call[0] == "observe" for call in lat.calls)
    assert any(call[0] == "inc" for call in err.calls)
    assert any(call[0] == "observe" for call in tps.calls)


@pytest.mark.asyncio
async def test_ensure_ollama_model_async_paths(monkeypatch):
    """Testa ensure ollama model async paths."""
    class _Resp:
        """Represent `_Resp` within this module.

The class groups the state and behavior required for Resp."""
        def __init__(self, code, payload=None):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self.status_code = code
            self._payload = payload or {}

        def json(self):
            """Execute the json routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return self._payload

        def raise_for_status(self):
            """Raise a synthetic HTTP error when the mocked status is unsuccessful."""
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("bad status", request=None, response=None)

    class _Client:
        """Represent `_Client` within this module.

The class groups the state and behavior required for Client."""
        def __init__(self, tags_resp, pull_resp):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self.tags_resp = tags_resp
            self.pull_resp = pull_resp

        async def get(self, *a, **k):
            """Execute the get routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return self.tags_resp

        async def post(self, *a, **k):
            """Execute the post routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return self.pull_resp

    async def _c1():
        """Execute the c1 routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return _Client(_Resp(200, {"models": [{"name": "phi4:latest"}]}), _Resp(200))

    monkeypatch.setattr(pa, "get_http_client", _c1)
    assert await pa._ensure_ollama_model_async("phi4:latest") is True

    async def _c2():
        """Execute the c2 routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return _Client(_Resp(200, {"models": []}), _Resp(200))

    monkeypatch.setattr(pa, "get_http_client", _c2)
    assert await pa._ensure_ollama_model_async("phi4:latest") is True

    async def _c3():
        """Execute the c3 routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return _Client(_Resp(500, {}), _Resp(500))

    monkeypatch.setattr(pa, "get_http_client", _c3)
    assert await pa._ensure_ollama_model_async("phi4:latest") is False

    async def _boom():
        """Execute the boom routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        raise httpx.ConnectError("x")

    monkeypatch.setattr(pa, "get_http_client", _boom)
    assert await pa._ensure_ollama_model_async("phi4:latest") is False


@pytest.mark.asyncio
async def test_fetch_ollama_tags_cache_and_probe(monkeypatch):
    """fetch_ollama_tags should reuse cached tags and expose a lightweight probe."""
    calls = {"get": 0, "head": 0}

    class _Resp:
        def __init__(self, payload=None):
            self._payload = payload or {}

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class _Client:
        async def get(self, *args, **kwargs):
            calls["get"] += 1
            return _Resp({"models": [{"name": "qwen3.5:2b"}]})

        async def head(self, *args, **kwargs):
            calls["head"] += 1
            return _Resp()

    async def _client():
        return _Client()

    pa.reset_ollama_tags_cache()
    monkeypatch.setattr(pa, "get_http_client", _client)
    tags_first = await pa.fetch_ollama_tags(force_refresh=True, ttl_seconds=60)
    tags_second = await pa.fetch_ollama_tags(force_refresh=False, ttl_seconds=60)
    await pa.probe_ollama_liveness()

    assert tags_first == tags_second
    assert calls["get"] == 1
    assert calls["head"] == 1
