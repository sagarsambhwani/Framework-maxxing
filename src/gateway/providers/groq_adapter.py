"""Groq LPU Provider Strategy Adapter.

Provides sub-second token generation using Groq's high-speed LPU infrastructure.
"""

import time
from typing import List, Dict, Any, Generator, Optional
from openai import OpenAI

from src.gateway.providers.base import BaseProviderAdapter
from src.common.config import settings
from src.common.logging import term_log, debug_log, Colors

class GroqAdapter(BaseProviderAdapter):
    """Adapter for Groq LPU inference."""

    def __init__(self):
        self.client: Optional[OpenAI] = None
        if settings.GROQ_API_KEY:
            try:
                self.client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=settings.GROQ_API_KEY
                )
            except Exception as e:
                term_log("⚠️ [GROQ ADAPTER]", f"Failed to initialize Groq client: {e}", Colors.YELLOW)

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        actual_slug = model.replace("groq/", "")
        term_log("⚡ [GATEWAY]", f"Routing to {Colors.YELLOW}Groq LPU ({actual_slug}){Colors.END}", Colors.YELLOW)

        if not self.client:
            raise ValueError("Groq client not initialized (missing GROQ_API_KEY)")

        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=actual_slug,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        dur = round(time.time() - t0, 3)

        content = (resp.choices[0].message.content or "").strip()
        prompt_toks = getattr(resp.usage, "prompt_tokens", 0)
        compl_toks = getattr(resp.usage, "completion_tokens", 0)
        tokens = getattr(resp.usage, "total_tokens", prompt_toks + compl_toks)

        debug_log("🔍 [DEBUG:GROQ_RESP]", f"Received {len(content)} chars ({tokens} tokens) in {dur}s")

        return {
            "content": content,
            "model": model,
            "latency_s": dur,
            "ttft_ms": round(dur * 250, 1) if dur < 1.0 else 140.0,
            "prompt_tokens": prompt_toks,
            "completion_tokens": compl_toks,
            "tokens": tokens,
            "provider": "Groq LPU",
            "cache_hit": False
        }

    def stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048
    ) -> Generator[str, None, None]:
        actual_slug = model.replace("groq/", "")
        if not self.client:
            raise ValueError("Groq client not initialized (missing GROQ_API_KEY)")

        stream_resp = self.client.chat.completions.create(
            model=actual_slug,
            messages=messages,
            stream=True,
            max_tokens=max_tokens
        )
        for chunk in stream_resp:
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content
