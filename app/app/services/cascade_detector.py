# Objective: Detect a cascading failure across models and pick an emergency route.
"""Cascade detection, split out of ``reliability.py``.

The detector answers one question — what fraction of the known models has an
open circuit breaker — and turns it into a severity level and, above a
threshold, an emergency model to route to. It shares nothing with the circuit
breaker registry or the request deduplicator beyond reading their state, so it
lives on its own; ``reliability`` re-exports it for every existing caller.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# Cascade Failure Detection
# ==============================================================================

class CascadeDetector:
    """
    Detects cascade failures across circuit breakers.

    Monitors the ratio of failed models and signals severity levels
    when too many models are failing simultaneously.
    """

    # Severity thresholds based on ratio of failed models
    THRESHOLDS = {
        "warning": 0.3,    # 30% models failed
        "critical": 0.5,   # 50% models failed
        "emergency": 0.8,  # 80% models failed
    }

    # Severity level enum
    SEVERITY_NORMAL = 0
    SEVERITY_WARNING = 1
    SEVERITY_CRITICAL = 2
    SEVERITY_EMERGENCY = 3

    _instance: Optional["CascadeDetector"] = None
    _lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> "CascadeDetector":
        """Return the process-wide cascade detector singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize emergency fallback configuration once for the singleton.

        Emergency fallback models are loaded lazily from dynamic settings so the
        detector can react to runtime configuration changes while still having a
        safe local default when settings are unavailable.
        """
        if self._initialized:
            return

        # Load emergency fallback models from settings
        try:
            from app.settings_dynamic import settings as dynamic_settings

            self._emergency_fallback_models = dynamic_settings.EMERGENCY_FALLBACK_MODELS
            if not self._emergency_fallback_models:
                # Default fallback if setting is empty
                self._emergency_fallback_models = [
                    "ollama/phi4:latest",
                    "ollama/gemma3:4b",
                    "ollama/llama3:8b",
                ]
        except Exception:
            self._emergency_fallback_models = [
                "ollama/phi4:latest",
                "ollama/gemma3:4b",
                "ollama/llama3:8b",
            ]
        self._initialized = True

    def get_failed_model_ratio(self) -> Tuple[float, int, int]:
        """
        Calculate the ratio of models with open circuit breakers.

        Returns:
            Tuple of (ratio, failed_count, total_count)
        """
        # Import diferido: reliability importa este módulo.
        from app.reliability import get_circuit_breaker_manager

        manager = get_circuit_breaker_manager()
        statuses = manager.get_all_statuses()

        if not statuses:
            return 0.0, 0, 0

        total = len(statuses)
        failed = sum(1 for s in statuses if s.get("state") == "open")

        ratio = failed / total if total > 0 else 0.0
        return ratio, failed, total

    def get_severity(self) -> int:
        """
        Get current cascade failure severity level.

        Returns:
            Severity level (0=normal, 1=warning, 2=critical, 3=emergency)
        """
        ratio, _, _ = self.get_failed_model_ratio()

        if ratio >= self.THRESHOLDS["emergency"]:
            return self.SEVERITY_EMERGENCY
        elif ratio >= self.THRESHOLDS["critical"]:
            return self.SEVERITY_CRITICAL
        elif ratio >= self.THRESHOLDS["warning"]:
            return self.SEVERITY_WARNING
        else:
            return self.SEVERITY_NORMAL

    def get_severity_name(self) -> str:
        """Get severity level as string."""
        severity = self.get_severity()
        names = {
            self.SEVERITY_NORMAL: "normal",
            self.SEVERITY_WARNING: "warning",
            self.SEVERITY_CRITICAL: "critical",
            self.SEVERITY_EMERGENCY: "emergency",
        }
        return names.get(severity, "unknown")

    @property
    def is_emergency_mode(self) -> bool:
        """Check if system is in emergency mode."""
        return self.get_severity() >= self.SEVERITY_EMERGENCY

    @property
    def is_degraded(self) -> bool:
        """Check if system is in any degraded state."""
        return self.get_severity() >= self.SEVERITY_WARNING

    def get_emergency_fallback(self) -> Optional[str]:
        """
        Get a fallback model for emergency routing.

        In emergency mode, returns a known-reliable local model.
        Returns None if no fallback is available.
        """
        if not self.is_emergency_mode:
            return None

        # Import diferido: reliability importa este módulo.
        from app.reliability import get_circuit_breaker_manager

        manager = get_circuit_breaker_manager()

        # Try emergency fallback models in order
        for model in self._emergency_fallback_models:
            if manager.is_available(model):
                return model

        # All fallbacks are down - return first one anyway
        # (will likely fail but maintains consistent behavior)
        return self._emergency_fallback_models[0] if self._emergency_fallback_models else None

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive cascade detection status."""
        ratio, failed, total = self.get_failed_model_ratio()
        severity = self.get_severity()

        return {
            "severity": severity,
            "severity_name": self.get_severity_name(),
            "failed_model_ratio": round(ratio, 3),
            "failed_models": failed,
            "total_models": total,
            "is_emergency_mode": self.is_emergency_mode,
            "is_degraded": self.is_degraded,
            "emergency_fallback": self.get_emergency_fallback() if self.is_emergency_mode else None,
            "thresholds": self.THRESHOLDS,
        }

    def check_and_log_warnings(self) -> Dict[str, Any]:
        """
        Check cascade status and log warnings if needed.

        Returns status dict with any warning messages.
        """
        status = self.get_status()
        warnings = []

        if status["severity"] == self.SEVERITY_EMERGENCY:
            msg = (
                f"🚨 CASCADE EMERGENCY: {status['failed_models']}/{status['total_models']} "
                f"models failing ({status['failed_model_ratio']:.0%})"
            )
            logger.critical(msg)
            warnings.append(msg)
        elif status["severity"] == self.SEVERITY_CRITICAL:
            msg = (
                f"⚠️ CASCADE CRITICAL: {status['failed_models']}/{status['total_models']} "
                f"models failing ({status['failed_model_ratio']:.0%})"
            )
            logger.error(msg)
            warnings.append(msg)
        elif status["severity"] == self.SEVERITY_WARNING:
            msg = (
                f"⚡ Cascade warning: {status['failed_models']}/{status['total_models']} "
                f"models failing ({status['failed_model_ratio']:.0%})"
            )
            logger.warning(msg)
            warnings.append(msg)

        status["warnings"] = warnings
        return status


def get_cascade_detector() -> CascadeDetector:
    """Get the global cascade detector instance."""
    return CascadeDetector()


