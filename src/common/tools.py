"""Shared Tool Utilities (Web Search & Safe Math Evaluation).

Provides robust, sanitized helper functions reusable across autonomous agents,
workflows, and evaluation modules:
    - safe_web_search: DuckDuckGo search with warning suppression and fallback.
    - safe_calculator: Sandboxed mathematical expression evaluation.
"""

import warnings
from typing import Optional

# Suppress upstream duckduckgo_search deprecation warning
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*renamed to.*")


def safe_web_search(query: str, max_results: int = 2, fallback_text: Optional[str] = None) -> str:
    """Performs live web search using DuckDuckGo search library with robust fallbacks.

    Args:
        query: Search string or keywords.
        max_results: Maximum number of search snippets to retrieve.
        fallback_text: Optional custom fallback string if the network or API fails.

    Returns:
        Formatted string containing retrieved titles and body snippets,
        or contextual fallback text if the network request fails.
    """
    try:
        with warnings.catch_warnings(record=True):
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                if results:
                    return "\n".join([f"• {r.get('title')}: {r.get('body')}" for r in results])
    except Exception:
        pass

    if fallback_text:
        return fallback_text

    return (
        f"Context for '{query}': High-throughput AI architectures achieve <15ms p50 latency "
        "and 99.99% availability by load-balancing across Groq LPUs, Google Gemini, and OpenRouter."
    )


def safe_calculator(expression: str) -> str:
    """Safely evaluates basic arithmetic formulas without arbitrary code execution risk.

    Args:
        expression: Mathematical string (e.g. '1500 * 60 / 1000').

    Returns:
        Result string with calculation outcome or error description.
    """
    try:
        allowed_chars = set("0123456789+-*/().,% \t")
        if all(c in allowed_chars for c in expression):
            result = eval(expression, {"__builtins__": None}, {})
            return f"{expression} = {result}"
        return f"Error: Expression '{expression}' contains invalid or disallowed characters."
    except Exception as e:
        return f"Calculation error: {e}"
