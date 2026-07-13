# -*- coding: utf-8 -*-
# Objective: Application runtime code for model registry.
"""Maintain the router's canonical catalog of supported model configurations.

The registry is the authoritative source for provider ownership, capability
flags, pricing metadata, timeout defaults, and fallback chains. Runtime code
uses it to answer two different questions:

- what the application knows how to support in principle
- what models are actually eligible right now given the current environment

That distinction matters because the codebase retains support for paid APIs
while local deployments may intentionally expose only providers that have valid
credentials or local runtimes configured.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_PROVIDER_ENV_KEYS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


# ==============================================================================
# Enums
# ==============================================================================

class Provider(str, Enum):
    """Enumerate the provider backends recognized by the registry."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


class Capability(str, Enum):
    """Describe functional traits used during routing and filtering."""
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
    """Describe one model's operational metadata and routing constraints."""
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
        """Normalize derived fields after dataclass initialization."""
        if not self.display_name:
            self.display_name = self.name

    @property
    def full_name(self) -> str:
        """Get the full model name with provider prefix."""
        return f"{self.provider.value}/{self.name}"

    @property
    def supports_vision(self) -> bool:
        """Return whether the model accepts visual inputs."""
        return Capability.VISION in self.capabilities

    @property
    def supports_streaming(self) -> bool:
        """Return whether the model supports streamed generation semantics."""
        return Capability.STREAMING in self.capabilities

    @property
    def supports_reasoning(self) -> bool:
        """Return whether the model is treated as a reasoning-oriented model."""
        return Capability.REASONING in self.capabilities

    @property
    def supports_tools(self) -> bool:
        """Return whether the model supports tool/function calling."""
        return Capability.FUNCTION_CALLING in self.capabilities

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate request cost from token counts using registry pricing data."""
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        return round(input_cost + output_cost, 6)


# ==============================================================================
# Model Registry
# ==============================================================================

class ModelRegistry:
    """Expose a singleton registry of model definitions and lookup helpers.

    The registry keeps model metadata in memory and offers convenience methods
    for filtering by capability, provider availability, cost, and fallback
    relationships. It intentionally behaves like a process-wide singleton
    because routing code expects a stable, shared catalog across requests.
    """

    _instance: Optional["ModelRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "ModelRegistry":
        """Return the process-wide singleton registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Populate the registry exactly once with the default model catalog."""
        if self._initialized:
            return

        self._models: Dict[str, ModelConfig] = {}
        self._initialize_default_models()
        self._initialized = True

    def _initialize_default_models(self):
        """Register the built-in model catalog used by the application."""

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

    @staticmethod
    def is_provider_configured(provider: Provider | str) -> bool:
        """Return whether one provider is configured for runtime use."""
        provider_value = provider.value if isinstance(provider, Provider) else str(provider)
        if provider_value == Provider.OLLAMA.value:
            return True

        env_keys = _PROVIDER_ENV_KEYS.get(provider_value, ())
        if provider_value == Provider.OPENROUTER.value:
            try:
                from app.openrouter_catalog import get_openrouter_api_key

                if get_openrouter_api_key():
                    return True
            except Exception:
                pass
        return any(os.getenv(key, "").strip() for key in env_keys)

    def is_model_configured(self, model_name: str) -> bool:
        """Return whether a model can be used with the current environment."""
        config = self.get(model_name)
        if config is not None:
            return self.is_provider_configured(config.provider)

        if "/" not in model_name:
            return True

        prefix = model_name.split("/", 1)[0]
        try:
            provider = Provider(prefix)
        except ValueError:
            return False
        return self.is_provider_configured(provider)

    def filter_configured_model_names(self, model_names: List[str]) -> List[str]:
        """Keep model names whose provider is configured in the environment."""
        return [name for name in model_names if self.is_model_configured(name)]

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
        name = model_name
        if "/" in model_name:
            prefix, remainder = model_name.split("/", 1)
            try:
                provider = Provider(prefix)
            except ValueError:
                remainder = model_name
            if provider == Provider.OPENROUTER:
                name = remainder
            else:
                name = model_name.split("/")[-1]

        return ModelConfig(
            name=name,
            provider=provider,
            default_timeout=60,
        )

    def list_models(
        self,
        provider: Optional[Provider] = None,
        capability: Optional[Capability] = None,
        configured_only: bool = True,
    ) -> List[ModelConfig]:
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
            if configured_only and not self.is_provider_configured(config.provider):
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
                        if fallback_config and self.is_model_configured(fallback):
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


def is_provider_configured(provider: Provider | str) -> bool:
    """Return whether one provider is configured for runtime use."""
    return get_model_registry().is_provider_configured(provider)


def is_model_configured(model_name: str) -> bool:
    """Return whether one model can be used with the current environment."""
    return get_model_registry().is_model_configured(model_name)


def filter_configured_model_names(model_names: List[str]) -> List[str]:
    """Keep only model names whose provider is configured for runtime use."""
    return get_model_registry().filter_configured_model_names(model_names)


# Helpers de capacidade (tools/visão) foram centralizados em model_capabilities.
# Reexportados aqui para compatibilidade com importadores existentes.
from app.model_capabilities import (  # noqa: E402,F401  (re-export)
    VISION_CAPABLE_MARKERS,
    VISION_ONLY_MARKERS,
    filter_tool_capable_model_names,
    filter_vision_capable_model_names,
    is_vision_only_model,
    model_supports_tools,
    model_supports_vision,
)
