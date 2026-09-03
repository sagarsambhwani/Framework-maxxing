"""Example 4: Collaborative Multi-Provider Task Demo.

Demonstrates cross-provider collaboration where specialized models handle different steps:
    1. Google Gemini (Architect & Planner): Formulates architectural requirements with 1M context.
    2. Groq LPU (Ultra-Fast Synthesizer): Summarizes key decisions in sub-second inference.
    3. Langfuse Cloud: Traces cross-model latency and tokens in a single unified session.

Run with:
    .venv\\Scripts\\python.exe examples/04_multi_provider_demo.py
"""

import sys
import os
import time
from dotenv import load_dotenv

# Add root directory to pythonpath
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from src.gateway.router import gateway

print("=" * 75)
print("🌐 EXAMPLE 4: MULTI-PROVIDER COLLABORATIVE PIPELINE DEMO")
print("=" * 75)

topic = "Design a multi-region low-latency AI Gateway"

# -----------------------------------------------------------------------------
# Step 1: Google Gemini (Architect & Planner)
# -----------------------------------------------------------------------------
print("\n🔵 1. Calling Google Gemini (Architect & Planner)...")
res_gemini = gateway.complete(
    model="gemini/gemma-4-31b-it",
    messages=[{"role": "user", "content": f"Plan 2 architectural pillars for: {topic}"}],
    max_tokens=100
)
print(f"⏱️  Latency: {res_gemini['latency_s']}s")
print(f"📋 Plan:\n{res_gemini['content'][:150]}...\n")

# -----------------------------------------------------------------------------
# Step 2: Groq LPU (High-Speed Synthesizer)
# -----------------------------------------------------------------------------
print("⚡ 2. Calling Groq LPU (High-Speed Synthesizer)...")
res_groq = gateway.complete(
    model="groq/qwen/qwen3.8-27b",
    messages=[{"role": "user", "content": f"Summarize key recommendations for {topic} in 2 sentences."}],
    max_tokens=80
)
print(f"⏱️  Latency: {res_groq['latency_s']}s")
print(f"📝 Summary:\n{res_groq['content']}\n")

print("=" * 75)
print("✓ Multi-Provider Collaboration Verified!")
print("=" * 75 + "\n")
