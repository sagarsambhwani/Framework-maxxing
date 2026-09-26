"""Abstract Base Provider Adapter (Strategy Pattern).

Defines the universal interface that all cloud provider adapters (Groq, Gemini, OpenRouter)
must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator

class BaseProviderAdapter(ABC):
    """Abstract interface for LLM provider strategies."""

    @abstractmethod
    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """Executes a synchronous completion and returns normalized dictionary."""
        pass

    @abstractmethod
    def stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048
    ) -> Generator[str, None, None]:
        """Yields text token chunks in real-time."""
        pass
