# Objective: Ollama concurrency/admission/VRAM runtime (roadmap #19).
"""Ollama-specific runtime extracted from providers_async: dynamic concurrency
controller, VRAM telemetry, admission snapshot, warm-model preferences and adaptive
timeouts. Depends on _infra; downstream of the provider classes."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import time
from typing import Any, Dict, List

import app.providers_async as _pa
from app.observability import (
    OLLAMA_DYNAMIC_CONCURRENCY_ADJUSTMENTS,
    OLLAMA_DYNAMIC_CONCURRENCY_TELEMETRY_FAILURES,
    OLLAMA_VRAM_TOTAL_BYTES,
    OLLAMA_VRAM_USED_BYTES,
    OLLAMA_VRAM_UTILIZATION_RATIO,
)

from ._infra import (
    DEFAULT_BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS,
    DEFAULT_OLLAMA_CONCURRENCY_LIMIT,
    DEFAULT_OLLAMA_VRAM_HIGH_WATERMARK,
    DEFAULT_OLLAMA_VRAM_TARGET_UTILIZATION,
    DEFAULT_PROVIDER_UNAVAILABLE_NEGATIVE_CACHE_TTL_SECONDS,
    OLLAMA_HOST,
    REASONING_MODEL_KEYWORDS,
    log_process_file_descriptor_limit,
)

logger = logging.getLogger("providers_async")


def _runtime_provider_settings():
    """Delegate to the facade so tests patching pa._runtime_provider_settings apply."""
    return _pa._runtime_provider_settings()


async def get_http_client():
    """Delegate to the facade so tests patching pa.get_http_client apply."""
    return await _pa.get_http_client()


def reset_state() -> None:
    """Reset Ollama-owned mutable runtime state (controller, in-flight, negative cache)."""
    _ollama_concurrency_controller._effective_limit = DEFAULT_OLLAMA_CONCURRENCY_LIMIT
    _ollama_concurrency_controller._last_limit = DEFAULT_OLLAMA_CONCURRENCY_LIMIT
    _ollama_concurrency_controller._latest_vram_used = 0
    _ollama_concurrency_controller._latest_vram_total = 0
    _ollama_concurrency_controller._latest_ratio = 0.0
    _ollama_concurrency_controller._telemetry_ok = False
    _ollama_concurrency_controller._stable_low_windows = 0
    _ollama_runtime_state.clear()
    _provider_unavailable_until.clear()


class OllamaConcurrencyController:
    """Track VRAM pressure and compute a conservative dynamic Ollama concurrency limit."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._effective_limit = DEFAULT_OLLAMA_CONCURRENCY_LIMIT
        self._last_limit = DEFAULT_OLLAMA_CONCURRENCY_LIMIT
        self._latest_vram_used = 0
        self._latest_vram_total = 0
        self._latest_ratio = 0.0
        self._telemetry_ok = False
        self._stable_low_windows = 0

    async def start(self) -> None:
        """Start the background polling loop if dynamic mode is enabled."""
        cfg = _runtime_provider_settings()
        fallback_limit = int(cfg["ollama_concurrency_limit"])
        self._effective_limit = fallback_limit
        self._last_limit = fallback_limit
        if not bool(cfg["ollama_dynamic_concurrency_enabled"]):
            self._publish_mode(dynamic=False)
            self._publish_limit(fallback_limit)
            return

        async with self._lock:
            if self._task and not self._task.done():
                return
            self._stop_event = asyncio.Event()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the background polling loop."""
        async with self._lock:
            if not self._task:
                return
            self._stop_event.set()
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def get_effective_limit(self) -> int:
        """Return the latest effective concurrency limit."""
        cfg = _runtime_provider_settings()
        fallback_limit = int(cfg["ollama_concurrency_limit"])
        if not bool(cfg["ollama_dynamic_concurrency_enabled"]):
            self._publish_mode(dynamic=False)
            self._publish_limit(fallback_limit)
            return fallback_limit
        return max(1, int(self._effective_limit or fallback_limit))

    async def force_refresh(self) -> int:
        """Run one telemetry refresh cycle synchronously and return the new limit."""
        await self._refresh_once()
        return self.get_effective_limit()

    async def _run(self) -> None:
        """Continuously refresh VRAM telemetry and the derived concurrency limit."""
        while not self._stop_event.is_set():
            await self._refresh_once()
            cfg = _runtime_provider_settings()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=float(cfg["ollama_vram_poll_interval_seconds"]),
                )
            except asyncio.TimeoutError:
                continue

    async def _refresh_once(self) -> None:
        """Read VRAM telemetry once and update the effective concurrency limit."""
        cfg = _runtime_provider_settings()
        fallback_limit = int(cfg["ollama_concurrency_limit"])
        if not bool(cfg["ollama_dynamic_concurrency_enabled"]):
            self._effective_limit = fallback_limit
            self._publish_mode(dynamic=False)
            self._publish_limit(fallback_limit)
            return

        reading = await asyncio.to_thread(self._read_vram_snapshot, int(cfg["ollama_gpu_index"]))
        if reading is None:
            OLLAMA_DYNAMIC_CONCURRENCY_TELEMETRY_FAILURES.inc()
            self._telemetry_ok = False
            self._stable_low_windows = 0
            self._effective_limit = self._effective_limit or fallback_limit
            self._publish_mode(dynamic=False)
            self._publish_limit(self._effective_limit)
            return

        used_bytes, total_bytes = reading
        ratio = (used_bytes / total_bytes) if total_bytes > 0 else 0.0
        self._latest_vram_used = used_bytes
        self._latest_vram_total = total_bytes
        self._latest_ratio = ratio
        self._telemetry_ok = True
        self._publish_mode(dynamic=True)
        self._publish_vram(used_bytes, total_bytes, ratio)

        current_limit = max(
            int(cfg["ollama_concurrency_min"]),
            min(int(cfg["ollama_concurrency_max"]), int(self._effective_limit or fallback_limit)),
        )
        high_watermark = float(cfg["ollama_vram_high_watermark"])
        low_watermark = float(cfg["ollama_vram_low_watermark"])
        step_up = int(cfg["ollama_concurrency_step_up"])
        step_down = int(cfg["ollama_concurrency_step_down"])
        stable_windows = int(cfg["ollama_concurrency_stable_windows"])
        min_limit = int(cfg["ollama_concurrency_min"])
        max_limit = min(int(cfg["ollama_concurrency_max"]), fallback_limit)

        new_limit = current_limit
        direction: str | None = None
        if ratio >= high_watermark:
            new_limit = max(min_limit, current_limit - step_down)
            self._stable_low_windows = 0
            direction = "down" if new_limit != current_limit else None
        elif ratio <= low_watermark:
            self._stable_low_windows += 1
            if self._stable_low_windows >= stable_windows:
                new_limit = min(max_limit, current_limit + step_up)
                self._stable_low_windows = 0
                direction = "up" if new_limit != current_limit else None
        else:
            self._stable_low_windows = 0

        self._effective_limit = new_limit
        self._publish_limit(new_limit)
        if direction:
            OLLAMA_DYNAMIC_CONCURRENCY_ADJUSTMENTS.labels(direction=direction).inc()
            logger.info(
                "[ollama] Dynamic concurrency adjusted from %s to %s (vram_used=%s, vram_total=%s, ratio=%.3f)",
                current_limit,
                new_limit,
                used_bytes,
                total_bytes,
                ratio,
            )

    def _read_vram_snapshot(self, gpu_index: int) -> tuple[int, int] | None:
        """Return one `(used_bytes, total_bytes)` snapshot from `nvidia-smi`."""
        cmd = [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            line = (proc.stdout or "").strip().splitlines()[0]
            used_mb_str, total_mb_str = [part.strip() for part in line.split(",", 1)]
            return int(used_mb_str) * 1024 * 1024, int(total_mb_str) * 1024 * 1024
        except Exception:
            return None

    def _publish_vram(self, used_bytes: int, total_bytes: int, ratio: float) -> None:
        """Publish current VRAM telemetry gauges."""
        try:
            OLLAMA_VRAM_USED_BYTES.set(used_bytes)
            OLLAMA_VRAM_TOTAL_BYTES.set(total_bytes)
            OLLAMA_VRAM_UTILIZATION_RATIO.set(ratio)
        except Exception:
            pass

    def _publish_limit(self, limit: int) -> None:
        """Publish current effective dynamic limit."""
        try:
            _pa.OLLAMA_DYNAMIC_CONCURRENCY_LIMIT.set(limit)
        except Exception:
            pass

    def _publish_mode(self, *, dynamic: bool) -> None:
        """Publish whether dynamic mode is active or static fallback is in effect."""
        try:
            _pa.OLLAMA_DYNAMIC_CONCURRENCY_MODE.set(1 if dynamic else 0)
        except Exception:
            pass


_ollama_concurrency_controller = OllamaConcurrencyController()
_ollama_runtime_state: Dict[str, Dict[str, Any]] = {}
_provider_unavailable_until: Dict[str, float] = {}


def _normalize_ollama_model_name(model_name: str) -> str:
    """Return one canonical Ollama model label with the `ollama/` namespace."""
    cleaned = str(model_name or "").strip()
    if not cleaned:
        return "ollama/unknown"
    return cleaned if cleaned.startswith("ollama/") else f"ollama/{cleaned}"


def _ensure_ollama_runtime_entry(model_name: str) -> Dict[str, Any]:
    """Return mutable runtime state for one Ollama model, creating defaults on first use."""
    normalized = _normalize_ollama_model_name(model_name)
    return _ollama_runtime_state.setdefault(
        normalized,
        {
            "loaded": False,
            "inflight": 0,
            "last_load_seconds": 0.0,
            "last_queue_wait_seconds": 0.0,
            "last_used_at": 0.0,
        },
    )


def _mark_ollama_model_state(
    model_name: str,
    *,
    loaded: bool | None = None,
    load_seconds: float | None = None,
    inflight_delta: int = 0,
    queue_wait_seconds: float | None = None,
) -> None:
    """Update one Ollama model runtime snapshot used for routing preferences."""
    state = _ensure_ollama_runtime_entry(model_name)
    if loaded is not None:
        state["loaded"] = bool(loaded)
    if load_seconds is not None:
        state["last_load_seconds"] = max(0.0, float(load_seconds))
    if inflight_delta:
        state["inflight"] = max(0, int(state.get("inflight", 0)) + inflight_delta)
    if queue_wait_seconds is not None:
        state["last_queue_wait_seconds"] = max(0.0, float(queue_wait_seconds))
    state["last_used_at"] = time.time()


def get_configured_ollama_warm_models() -> List[str]:
    """Return the configured warm-model list normalized to the Ollama namespace."""
    cfg = _runtime_provider_settings()
    return list(cfg.get("ollama_warm_models", []) or [])


def get_interactive_ollama_models() -> List[str]:
    """Return the preferred interactive warm-model set, falling back to the global warm list."""
    cfg = _runtime_provider_settings()
    interactive = list(cfg.get("ollama_interactive_warm_models", []) or [])
    return interactive or get_configured_ollama_warm_models()


def apply_ollama_performance_preferences(
    candidates: List[str], runtime_hints: Dict[str, Any] | None = None
) -> List[str]:
    """Reorder and trim local candidates so routing prefers warm, less-saturated Ollama models."""
    cfg = _runtime_provider_settings()
    if not bool(cfg["ollama_load_penalty_enabled"]):
        return list(candidates)

    local_candidates = [candidate for candidate in candidates if str(candidate).startswith("ollama/")]
    if len(local_candidates) <= 1:
        return list(candidates)

    warm_set = set(get_configured_ollama_warm_models())
    interactive_warm_set = set(get_interactive_ollama_models())
    route_limit = max(2, int(cfg["ollama_route_candidate_limit"]))
    current_limit = max(1, int(_ollama_concurrency_controller.get_effective_limit()))
    workload_class = str((runtime_hints or {}).get("workload_class", "") or "").strip()
    detected_complexity = str((runtime_hints or {}).get("detected_complexity", "") or "").strip()
    interactive_priority = str((runtime_hints or {}).get("interactive_priority", "") or "").strip()

    def _score(candidate: str) -> tuple[float, int]:
        state = _ensure_ollama_runtime_entry(candidate)
        inflight = int(state.get("inflight", 0))
        loaded = bool(state.get("loaded", False))
        last_load_seconds = float(state.get("last_load_seconds", 0.0) or 0.0)
        last_queue_wait_seconds = float(state.get("last_queue_wait_seconds", 0.0) or 0.0)
        penalty = 0.0
        if not loaded:
            penalty += 7.0
        if candidate not in warm_set:
            penalty += 0.5
        if workload_class in {"simple_text", "knowledge_lookup"} and candidate not in interactive_warm_set:
            penalty += 2.5
        if detected_complexity in {"high", "expert"} and candidate.startswith("ollama/"):
            penalty += 5.0
        if interactive_priority == "high" and candidate in interactive_warm_set:
            penalty -= 1.5
        penalty += float(inflight) * 2.0
        if inflight >= max(1, current_limit - 1):
            penalty += 3.0
        penalty += min(4.0, last_queue_wait_seconds * 2.0)
        penalty += min(3.0, last_load_seconds / 5.0)
        return penalty, candidates.index(candidate)

    preferred_locals = sorted(local_candidates, key=_score)[: min(route_limit, len(local_candidates))]
    remaining = [candidate for candidate in candidates if not candidate.startswith("ollama/")]
    return preferred_locals + remaining


async def warm_ollama_model_runtime(model_name: str) -> bool:
    """Issue a tiny generate request so one local model is resident before user traffic arrives."""
    normalized = _normalize_ollama_model_name(model_name)
    short_name = normalized.split("/", 1)[1]
    try:
        client = await get_http_client()
        started_at = time.time()
        resp = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": short_name,
                "prompt": "Reply only with OK.",
                "stream": False,
                "think": False,
                "keep_alive": "15m",
                "options": {
                    "temperature": 0.0,
                    "num_predict": 4,
                    "num_ctx": 512,
                },
            },
            timeout=max(30.0, _get_adaptive_timeout(short_name)),
        )
        resp.raise_for_status()
        data = resp.json()
        load_seconds = float(data.get("load_duration", 0) or 0.0) / 1_000_000_000.0
        _mark_ollama_model_state(normalized, loaded=True, load_seconds=load_seconds)
        try:
            _pa.OLLAMA_MODEL_LOADED.labels(model=short_name).set(1)
            if load_seconds > 0:
                _pa.OLLAMA_MODEL_LOAD_SECONDS.labels(model=short_name).observe(load_seconds)
        except Exception:
            pass
        logger.info("[ollama] Warmed model %s in %.3fs", normalized, time.time() - started_at)
        return True
    except Exception as exc:
        logger.warning("[ollama] Failed to warm model %s: %s", normalized, exc)
        return False


async def start_provider_runtime_services() -> None:
    """Start background provider-side runtime services."""
    log_process_file_descriptor_limit("provider-runtime")
    await _ollama_concurrency_controller.start()


async def stop_provider_runtime_services() -> None:
    """Stop background provider-side runtime services."""
    await _ollama_concurrency_controller.stop()


def mark_provider_unavailable(model_name: str, *, ttl_seconds: int | None = None) -> None:
    """Remember a temporarily unavailable model to avoid hammering the same failing path."""
    cfg = _runtime_provider_settings()
    ttl = int(
        ttl_seconds
        if ttl_seconds is not None
        else cfg.get(
            "provider_unavailable_negative_cache_ttl_seconds",
            cfg.get(
                "model_unavailable_negative_cache_ttl_seconds", DEFAULT_PROVIDER_UNAVAILABLE_NEGATIVE_CACHE_TTL_SECONDS
            ),
        )
    )
    if ttl <= 0:
        return
    _provider_unavailable_until[str(model_name)] = time.time() + ttl


def clear_provider_unavailable(model_name: str) -> None:
    """Clear temporary negative-cache state after a successful call."""
    _provider_unavailable_until.pop(str(model_name), None)


def is_provider_temporarily_unavailable(model_name: str) -> bool:
    """Return whether one model is still inside the negative-cache cooldown window."""
    expires_at = float(_provider_unavailable_until.get(str(model_name), 0.0) or 0.0)
    if expires_at <= 0:
        return False
    if expires_at <= time.time():
        _provider_unavailable_until.pop(str(model_name), None)
        return False
    return True


def should_throttle_background_judge() -> bool:
    """Return whether background judge work should yield to interactive Ollama traffic."""
    cfg = _runtime_provider_settings()
    if not bool(cfg["ollama_background_load_shedding_enabled"]):
        return False
    current_limit = max(1, int(_ollama_concurrency_controller.get_effective_limit()))
    total_inflight = sum(int(state.get("inflight", 0) or 0) for state in _ollama_runtime_state.values())
    inflight_threshold = max(
        1, min(current_limit, int(cfg.get("background_throttle_inflight_threshold", current_limit)))
    )
    if total_inflight >= inflight_threshold:
        return True
    if any(
        float(state.get("last_queue_wait_seconds", 0.0) or 0.0) * 1000.0
        >= float(cfg.get("background_throttle_queue_wait_p95_ms", DEFAULT_BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS))
        for state in _ollama_runtime_state.values()
    ):
        return True
    ratio = float(getattr(_ollama_concurrency_controller, "_latest_ratio", 0.0) or 0.0)
    if ratio >= float(cfg["ollama_vram_high_watermark"]):
        return True
    return False


def get_ollama_admission_snapshot() -> Dict[str, float | int | str]:
    """Return a compact snapshot of Ollama pressure for adaptive admission control.

    The middleware layer cannot read Prometheus samples directly, so it uses this
    runtime snapshot derived from the same in-process state that powers provider
    throttling and warm-model selection.
    """
    cfg = _runtime_provider_settings()
    current_limit = max(1, int(_ollama_concurrency_controller.get_effective_limit()))
    total_inflight = sum(int(state.get("inflight", 0) or 0) for state in _ollama_runtime_state.values())
    max_queue_wait_ms = 0.0
    for state in _ollama_runtime_state.values():
        max_queue_wait_ms = max(max_queue_wait_ms, float(state.get("last_queue_wait_seconds", 0.0) or 0.0) * 1000.0)
    utilization = min(2.0, float(total_inflight) / float(current_limit)) if current_limit > 0 else 0.0
    vram_ratio = float(getattr(_ollama_concurrency_controller, "_latest_ratio", 0.0) or 0.0)
    elevated_utilization = float(cfg.get("ollama_vram_target_utilization", DEFAULT_OLLAMA_VRAM_TARGET_UTILIZATION))
    congested_utilization = float(cfg.get("ollama_vram_high_watermark", DEFAULT_OLLAMA_VRAM_HIGH_WATERMARK))
    pressure_state = "normal"
    if (
        utilization >= 1.0
        or max_queue_wait_ms
        >= float(cfg.get("background_throttle_queue_wait_p95_ms", DEFAULT_BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS))
        or vram_ratio >= congested_utilization
    ):
        pressure_state = "congested"
    elif utilization >= elevated_utilization or max_queue_wait_ms >= max(
        100.0,
        float(cfg.get("background_throttle_queue_wait_p95_ms", DEFAULT_BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS)) * 0.5,
    ):
        pressure_state = "elevated"
    return {
        "current_limit": current_limit,
        "total_inflight": total_inflight,
        "max_queue_wait_ms": round(max_queue_wait_ms, 3),
        "utilization": round(utilization, 4),
        "vram_ratio": round(vram_ratio, 4),
        "pressure_state": pressure_state,
    }


def _workload_provider_timeout_budget(workload_class: str | None) -> float | None:
    """Return the configured timeout budget for a known workload class."""
    workload = str(workload_class or "").strip().lower()
    key_map = {
        "simple_text": ("PROVIDER_TIMEOUT_SIMPLE_SECONDS", 20),
        "knowledge_lookup": ("PROVIDER_TIMEOUT_KNOWLEDGE_SECONDS", 35),
        "reasoning": ("PROVIDER_TIMEOUT_REASONING_SECONDS", 60),
        "vision": ("PROVIDER_TIMEOUT_VISION_SECONDS", 90),
    }
    key_default = key_map.get(workload)
    if not key_default:
        return None
    key, default = key_default
    try:
        from app.settings_dynamic import settings as dynamic_settings

        return max(5.0, float(dynamic_settings.get(key, default)))
    except Exception:
        return float(default)


def _get_adaptive_timeout(model: str, workload_class: str | None = None) -> float:
    """
    Calculate a timeout budget that reflects the expected cost of the model call.

    Reasoning and larger models receive a larger timeout budget so normal
    behavior does not look like failure, while simpler models keep lower limits
    to surface stalls earlier.
    """
    explicit_workload_timeout = _workload_provider_timeout_budget(workload_class)
    if explicit_workload_timeout is not None:
        return explicit_workload_timeout

    runtime_cfg = _runtime_provider_settings()
    adaptive_timeout_enabled = bool(runtime_cfg["adaptive_timeout_enabled"])
    base_timeout = int(runtime_cfg["base_timeout"])
    max_timeout = int(runtime_cfg["max_timeout"])
    timeout_multiplier = float(runtime_cfg["timeout_multiplier"])
    reasoning_multiplier = float(runtime_cfg["reasoning_multiplier"])

    if not adaptive_timeout_enabled:
        return max_timeout  # Fall back to original behavior

    model_lower = model.lower()

    # Reasoning models need more time for chain-of-thought
    if any(k in model_lower for k in REASONING_MODEL_KEYWORDS):
        timeout = base_timeout * reasoning_multiplier
    # Large models (>7B parameters indicated by name)
    elif any(size in model_lower for size in ["70b", "65b", "33b", "13b", "11b"]):
        timeout = base_timeout * timeout_multiplier
    # Standard models
    else:
        timeout = base_timeout * 1.5

    return min(timeout, max_timeout)
