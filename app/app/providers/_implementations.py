# Objective: Concrete provider adapters (OpenAI/OpenRouter/Anthropic/Gemini/Ollama) (roadmap #19).
"""Re-export of the provider adapters, one module per provider.

- ``providers._base``: LLMResponse, BaseProvider and the facade shims that route
  test-patched hot symbols (get_http_client, get_model_cost, count_tokens,
  _runtime_provider_settings) through ``app.providers_async``.
- ``providers._openai``, ``_anthropic``, ``_gemini``, ``_ollama_adapter``.
"""

from ._anthropic import AnthropicProvider  # noqa: F401
from ._base import (  # noqa: F401
    BaseProvider,
    LLMResponse,
    _runtime_provider_settings,
    count_tokens,
    get_http_client,
    get_model_cost,
    logger,
)
from ._gemini import GeminiProvider  # noqa: F401
from ._ollama_adapter import OllamaProvider  # noqa: F401
from ._openai import OpenAIProvider, OpenRouterProvider  # noqa: F401
