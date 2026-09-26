"""Provider Adapters Package (Strategy & Factory Patterns).

Exposes provider strategy adapters and factory lookup function.
"""

from src.gateway.providers.base import BaseProviderAdapter
from src.gateway.providers.groq_adapter import GroqAdapter
from src.gateway.providers.gemini_adapter import GeminiAdapter
from src.gateway.providers.openrouter_adapter import OpenRouterAdapter

# Singleton instances
groq_adapter = GroqAdapter()
gemini_adapter = GeminiAdapter()
openrouter_adapter = OpenRouterAdapter()

def get_provider_adapter(model_slug: str) -> BaseProviderAdapter:
    """Factory method returning the appropriate provider strategy for a model slug."""
    if model_slug.startswith("groq/"):
        return groq_adapter
    elif model_slug.startswith("gemini/"):
        return gemini_adapter
    else:
        return openrouter_adapter

__all__ = [
    "BaseProviderAdapter",
    "GroqAdapter",
    "GeminiAdapter",
    "OpenRouterAdapter",
    "get_provider_adapter",
    "groq_adapter",
    "gemini_adapter",
    "openrouter_adapter"
]
