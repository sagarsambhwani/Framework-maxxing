"""Google Gemini Provider Strategy Adapter.

Provides massive context window (up to 1M tokens) and deep reasoning using Google Gemini.
"""

import time
from typing import List, Dict, Any, Generator
import litellm

from src.gateway.providers.base import BaseProviderAdapter
from src.common.config import settings
from src.common.logging import term_log, debug_log, Colors

class GeminiAdapter(BaseProviderAdapter):
    """Adapter for Google Gemini inference."""

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        term_log("🔵 [GATEWAY]", f"Routing to {Colors.CYAN}Google Gemini ({model}){Colors.END}", Colors.CYAN)
        t0 = time.time()
        resp = litellm.completion(
            model=model,
            messages=messages,
            api_key=settings.GEMINI_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens
        )
        dur = round(time.time() - t0, 3)

        content = (getattr(resp.choices[0].message, "content", "") or "").strip()
        prompt_toks = getattr(resp.usage, "prompt_tokens", 0)
        compl_toks = getattr(resp.usage, "completion_tokens", 0)
        tokens = getattr(resp.usage, "total_tokens", prompt_toks + compl_toks)

        debug_log("🔍 [DEBUG:GEMINI_RESP]", f"Received {len(content)} chars ({tokens} tokens) in {dur}s")

        return {
            "content": content,
            "model": model,
            "latency_s": dur,
            "ttft_ms": 650.0,
            "prompt_tokens": prompt_toks,
            "completion_tokens": compl_toks,
            "tokens": tokens,
            "provider": "Google Gemini",
            "cache_hit": False
        }

    def stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048
    ) -> Generator[str, None, None]:
        resp = litellm.completion(
            model=model,
            messages=messages,
            api_key=settings.GEMINI_API_KEY,
            stream=True,
            max_tokens=max_tokens
        )
        for chunk in resp:
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content
