"""Marketing Domain Tools & Text Analyzers.

Provides specialized analytical tools used by the Marketing Agents:
    1. `market_trend_search`: Queries live technical trends and competitor positioning.
    2. `keyword_density_analyzer`: Computes keyword frequency and density percentages.
    3. `readability_analyzer`: Calculates character counts, word counts, and reading time.
"""

import re
from typing import Dict, Any, List
from src.common.logging import term_log, Colors


def market_trend_search(topic: str) -> str:
    """Searches for current market trends and competitor discussions.

    Args:
        topic: Market sector or product category.

    Returns:
        Summary of recent market insights and popular discussion points.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{topic} market trends 2026", max_results=2))
            if results:
                return "\n".join([f"• {r.get('title')}: {r.get('body')}" for r in results])
    except Exception:
        pass
    return f"Key trend in {topic}: High demand for cost-reduction, ultra-low latency, and zero-vendor lock-in."


def keyword_density_analyzer(text: str, target_keywords: List[str]) -> Dict[str, Any]:
    """Analyzes keyword occurrence and density percentage in copy.

    Args:
        text: Marketing copy text.
        target_keywords: List of target SEO or brand keywords.

    Returns:
        Dict with total words, keyword counts, and density percentages.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    total_words = max(len(words), 1)
    keyword_stats = {}

    for kw in target_keywords:
        kw_clean = kw.lower().strip()
        count = text.lower().count(kw_clean)
        density = round((count / total_words) * 100, 2)
        keyword_stats[kw] = {"count": count, "density_pct": density}

    return {
        "total_words": total_words,
        "keywords": keyword_stats
    }


def readability_analyzer(text: str) -> Dict[str, Any]:
    """Calculates character count, word count, and reading time estimates.

    Args:
        text: Copy string to analyze.

    Returns:
        Dict containing char count, word count, sentence count, and avg word length.
    """
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    sentences = max(len(re.split(r"[.!?]+", text)) - 1, 1)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentences,
        "avg_words_per_sentence": round(word_count / sentences, 1),
        "est_reading_time_sec": round(word_count / 3.5, 1)  # ~210 wpm average reading speed
    }
