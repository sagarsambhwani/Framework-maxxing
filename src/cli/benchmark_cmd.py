"""Benchmark CLI Command Handler."""

from src.common.config import settings
from src.common.logging import print_banner


def run_benchmark():
    """Executes comparative latency benchmarks across all supported cloud providers."""
    from src.gateway.router import gateway

    print_banner(
        "MULTI-PROVIDER SPEED & LATENCY BENCHMARK",
        f"Testing Groq LPU vs Google Gemini vs OpenRouter | Debug Mode: {'ON' if settings.DEBUG_MODE else 'OFF'}"
    )

    benchmarks = [
        ("groq/qwen/qwen3.8-27b", "⚡ Groq LPU (Qwen 3.8 27B)"),
        ("gemini/gemma-4-31b-it", "🔵 Google Gemini (Gemma 31B)"),
        ("openrouter/inclusionai/ling-3.0-flash-fin:free", "🟢 OpenRouter (Ling 3.0 Flash)")
    ]

    for model_slug, label in benchmarks:
        res = gateway.complete(
            model=model_slug,
            messages=[{"role": "user", "content": "Explain token latency in 1 sentence."}],
            max_tokens=40
        )
        print(f"• {label:<35} : {res['latency_s']}s (Tokens: {res['tokens']})")
        print(f"  -> '{res['content'][:70]}...'\n")

    print("✓ Benchmark complete!")
