# -*- coding: utf-8 -*-
"""
model_registry.py — Centralized Model Configuration Registry
-------------------------------------------------------------
Provides a single source of truth for all model configurations including:
- Provider information
- Capabilities (vision, streaming, function calling)
- Pricing information
- Timeout and context limits
- Fallback chains for graceful degradation
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ==============================================================================
# Enums
# ==============================================================================

class Provider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class Capability(str, Enum):
    """Model capabilities."""
    TEXT = "text"
    VISION = "vision"
    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    REASONING = "reasoning"  # For models like o1, DeepSeek-R1, Phi-4


# ==============================================================================
# Model Configuration
# ==============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    provider: Provider
    display_name: str = ""

    # Capabilities
    capabilities: Set[Capability] = field(default_factory=lambda: {Capability.TEXT})

    # Context and tokens
    max_context_tokens: int = 4096
    max_output_tokens: int = 4096
    default_max_tokens: int = 512

    # Pricing (USD per 1K tokens)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    # Timeouts (seconds)
    default_timeout: int = 60
    max_timeout: int = 300

    # Reliability
    priority: int = 100  # Lower = higher priority for selection
    fallback_models: List[str] = field(default_factory=list)

    # Circuit breaker settings
    circuit_breaker_threshold: int = 5  # Failures before opening
    circuit_breaker_timeout: int = 60  # Seconds before half-open

    # Model-specific settings
    supports_system_prompt: bool = True
    requires_alternating_roles: bool = False

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name

    @property
    def full_name(self) -> str:
        """Get the full model name with provider prefix."""
        return f"{self.provider.value}/{self.name}"

    @property
    def supports_vision(self) -> bool:
        return Capability.VISION in self.capabilities

    @property
    def supports_streaming(self) -> bool:
        return Capability.STREAMING in self.capabilities

    @property
    def supports_reasoning(self) -> bool:
        return Capability.REASONING in self.capabilities

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for a request."""
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        return round(input_cost + output_cost, 6)


# ==============================================================================
# Model Registry
# ==============================================================================

class ModelRegistry:
    """
    Centralized registry for all model configurations.

    Usage:
        registry = ModelRegistry()
        config = registry.get("openai/gpt-4o")
        if config.supports_vision:
            # Handle vision request
    """

    _instance: Optional["ModelRegistry"] = None

    def __new__(cls) -> "ModelRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._models: Dict[str, ModelConfig] = {}
        self._initialize_default_models()
        self._initialized = True

    def _initialize_default_models(self):
        """Initialize with default model configurations."""

        # ==========================================================
        # OpenAI Models
        # ==========================================================
        self.register(ModelConfig(
            name="gpt-4o",
            provider=Provider.OPENAI,
            display_name="GPT-4o",
            capabilities={Capability.TEXT, Capability.VISION, Capability.STREAMING, Capability.FUNCTION_CALLING},
            max_context_tokens=128000,
            max_output_tokens=16384,
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            default_timeout=60,
            priority=10,
            fallback_models=["anthropic/claude-3-5-sonnet-latest", "gemini/gemini-1.5-pro"],
        ))

        self.register(ModelConfig(
            name="gpt-4o-mini",
            provider=Provider.OPENAI,
            display_name="GPT-4o Mini",
            capabilities={Capability.TEXT, Capability.VISION, Capability.STREAMING, Capability.FUNCTION_CALLING},
            max_context_tokens=128000,
            max_output_tokens=16384,
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            default_timeout=30,
            priority=20,
            fallback_models=["gemini/gemini-1.5-flash", "ollama/phi4:latest"],
        ))

        self.register(ModelConfig(
            name="o1",
            provider=Provider.OPENAI,
            display_name="o1 (Reasoning)",
            capabilities={Capability.TEXT, Capability.REASONING},
            max_context_tokens=200000,
            max_output_tokens=100000,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.060,
            default_timeout=120,
            max_timeout=600,
            priority=5,
            fallback_models=["openai/gpt-4o"],
        ))

        self.register(ModelConfig(
            name="o1-mini",
            provider=Provider.OPENAI,
            display_name="o1-mini (Reasoning)",
            capabilities={Capability.TEXT, Capability.REASONING},
            max_context_tokens=128000,
            max_output_tokens=65536,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.012,
            default_timeout=90,
            priority=15,
            fallback_models=["openai/gpt-4o-mini"],
        ))

        # ==========================================================
        # Anthropic Models
        # ==========================================================
        self.register(ModelConfig(
            name="claude-3-5-sonnet-latest",
            provider=Provider.ANTHROPIC,
            display_name="Claude 3.5 Sonnet",
            capabilities={Capability.TEXT, Capability.VISION, Capability.STREAMING, Capability.FUNCTION_CALLING},
            max_context_tokens=200000,
            max_output_tokens=8192,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            default_timeout=60,
            priority=10,
            fallback_models=["openai/gpt-4o", "gemini/gemini-1.5-pro"],
        ))

        self.register(ModelConfig(
            name="claude-3-5-haiku-latest",
            provider=Provider.ANTHROPIC,
            display_name="Claude 3.5 Haiku",
            capabilities={Capability.TEXT, Capability.VISION, Capability.STREAMING},
            max_context_tokens=200000,
            max_output_tokens=8192,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.005,
            default_timeout=30,
            priority=25,
            fallback_models=["openai/gpt-4o-mini", "gemini/gemini-1.5-flash"],
        ))

        self.register(ModelConfig(
            name="claude-3-opus-latest",
            provider=Provider.ANTHROPIC,
            display_name="Claude 3 Opus",
            capabilities={Capability.TEXT, Capability.VISION, Capability.STREAMING, Capability.FUNCTION_CALLING},
            max_context_tokens=200000,
            max_output_tokens=4096,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            default_timeout=90,
            priority=5,
            fallback_models=["openai/o1", "anthropic/claude-3-5-sonnet-latest"],
        ))

        # ==========================================================
        # Google Gemini Models
        # ==========================================================
        self.register(ModelConfig(
            name="gemini-1.5-pro",
            provider=Provider.GEMINI,
            display_name="Gemini 1.5 Pro",
            capabilities={Capability.TEXT, Capability.VISION, Capability.STREAMING, Capability.FUNCTION_CALLING},
            max_context_tokens=2000000,
            max_output_tokens=8192,
            cost_per_1k_input=0.00125,
            cost_per_1k_output=0.005,
            default_timeout=60,
            priority=15,
            fallback_models=["openai/gpt-4o", "anthropic/claude-3-5-sonnet-latest"],
        ))

        self.register(ModelConfig(
            name="gemini-1.5-flash",
            provider=Provider.GEMINI,
            display_name="Gemini 1.5 Flash",
            capabilities={Capability.TEXT, Capability.VISION, Capability.STREAMING},
            max_context_tokens=1000000,
            max_output_tokens=8192,
            cost_per_1k_input=0.000075,
            cost_per_1k_output=0.0003,
            default_timeout=30,
            priority=30,
            fallback_models=["openai/gpt-4o-mini", "ollama/phi4:latest"],
        ))

        self.register(ModelConfig(
            name="gemini-2.0-flash-exp",
            provider=Provider.GEMINI,
            display_name="Gemini 2.0 Flash (Experimental)",
            capabilities={Capability.TEXT, Capability.VISION, Capability.STREAMING, Capability.REASONING},
            max_context_tokens=1000000,
            max_output_tokens=8192,
            cost_per_1k_input=0.0,  # Free during preview
            cost_per_1k_output=0.0,
            default_timeout=45,
            priority=20,
            fallback_models=["gemini/gemini-1.5-flash"],
        ))

        # ==========================================================
        # Ollama (Local) Models
        # ==========================================================
        self.register(ModelConfig(
            name="phi4:latest",
            provider=Provider.OLLAMA,
            display_name="Phi-4 (Local)",
            capabilities={Capability.TEXT, Capability.REASONING},
            max_context_tokens=16384,
            max_output_tokens=4096,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            default_timeout=120,
            max_timeout=600,
            priority=50,
            fallback_models=["ollama/llama3.2:latest", "gemini/gemini-1.5-flash"],
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=30,
        ))

        self.register(ModelConfig(
            name="llama3.2:latest",
            provider=Provider.OLLAMA,
            display_name="Llama 3.2 (Local)",
            capabilities={Capability.TEXT},
            max_context_tokens=128000,
            max_output_tokens=4096,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            default_timeout=90,
            max_timeout=300,
            priority=55,
            fallback_models=["ollama/phi4:latest"],
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=30,
        ))

        self.register(ModelConfig(
            name="llava:latest",
            provider=Provider.OLLAMA,
            display_name="LLaVA (Local Vision)",
            capabilities={Capability.TEXT, Capability.VISION},
            max_context_tokens=4096,
            max_output_tokens=2048,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            default_timeout=120,
            max_timeout=300,
            priority=60,
            fallback_models=["gemini/gemini-1.5-flash", "openai/gpt-4o-mini"],
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=30,
        ))

        self.register(ModelConfig(
            name="deepseek-r1:latest",
            provider=Provider.OLLAMA,
            display_name="DeepSeek R1 (Local Reasoning)",
            capabilities={Capability.TEXT, Capability.REASONING},
            max_context_tokens=64000,
            max_output_tokens=8192,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            default_timeout=180,
            max_timeout=900,
            priority=45,
            fallback_models=["ollama/phi4:latest", "openai/o1-mini"],
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=30,
        ))

        logger.info(f"[ModelRegistry] Initialized with {len(self._models)} models")

    def register(self, config: ModelConfig) -> None:
        """Register a model configuration."""
        full_name = config.full_name
        self._models[full_name] = config
        # Also register by short name for convenience
        self._models[config.name] = config

    def get(self, model_name: str) -> Optional[ModelConfig]:
        """
        Get model configuration by name.

        Args:
            model_name: Full name (e.g., "openai/gpt-4o") or short name (e.g., "gpt-4o")

        Returns:
            ModelConfig or None if not found
        """
        return self._models.get(model_name)

    def get_or_default(self, model_name: str) -> ModelConfig:
        """Get model config or return a default configuration."""
        config = self.get(model_name)
        if config is not None:
            return config

        # Create a default config for unknown models
        provider = Provider.OLLAMA
        if "/" in model_name:
            prefix = model_name.split("/")[0]
            try:
                provider = Provider(prefix)
            except ValueError:
                pass

        return ModelConfig(
            name=model_name.split("/")[-1] if "/" in model_name else model_name,
            provider=provider,
            default_timeout=60,
        )

    def list_models(self, provider: Optional[Provider] = None, capability: Optional[Capability] = None) -> List[ModelConfig]:
        """
        List all registered models, optionally filtered.

        Args:
            provider: Filter by provider
            capability: Filter by capability

        Returns:
            List of matching ModelConfig objects
        """
        # Get unique configs (avoid duplicates from short name aliases)
        seen_names = set()
        models = []

        for config in self._models.values():
            if config.full_name in seen_names:
                continue
            seen_names.add(config.full_name)

            if provider and config.provider != provider:
                continue
            if capability and capability not in config.capabilities:
                continue

            models.append(config)

        return sorted(models, key=lambda m: m.priority)

    def get_fallback_chain(self, model_name: str, max_depth: int = 3) -> List[ModelConfig]:
        """
        Get the fallback chain for a model.

        Args:
            model_name: Starting model name
            max_depth: Maximum fallback depth

        Returns:
            List of ModelConfig in fallback order (excluding the original)
        """
        chain = []
        visited = {model_name}
        current_models = [model_name]

        for _ in range(max_depth):
            next_models = []
            for name in current_models:
                config = self.get(name)
                if config is None:
                    continue

                for fallback in config.fallback_models:
                    if fallback not in visited:
                        visited.add(fallback)
                        fallback_config = self.get(fallback)
                        if fallback_config:
                            chain.append(fallback_config)
                            next_models.append(fallback)

            if not next_models:
                break
            current_models = next_models

        return chain

    def get_vision_models(self) -> List[ModelConfig]:
        """Get all models that support vision."""
        return self.list_models(capability=Capability.VISION)

    def get_reasoning_models(self) -> List[ModelConfig]:
        """Get all models that support reasoning/chain-of-thought."""
        return self.list_models(capability=Capability.REASONING)

    def get_local_models(self) -> List[ModelConfig]:
        """Get all local (Ollama) models."""
        return self.list_models(provider=Provider.OLLAMA)

    def get_cheapest_model(self, capability: Optional[Capability] = None) -> Optional[ModelConfig]:
        """Get the cheapest model (by output cost), optionally with a specific capability."""
        models = self.list_models(capability=capability)
        if not models:
            return None

        # Filter out free models first, then find cheapest
        paid_models = [m for m in models if m.cost_per_1k_output > 0]
        if paid_models:
            return min(paid_models, key=lambda m: m.cost_per_1k_output)

        # If all are free, return highest priority
        return models[0]


# ==============================================================================
# Global Instance
# ==============================================================================

def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return ModelRegistry()


# Convenience function
def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get model configuration by name."""
    return get_model_registry().get(model_name)
