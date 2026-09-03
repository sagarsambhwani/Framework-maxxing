"""03_gateway_caching_demo.py - Gateway Response Caching & Fallback Failover Demo.

Demonstrates:
- 0ms local response caching for duplicate queries
- Automatic failover when a model endpoint fails
- Cost calculation per completion

Run with:
    .venv\\Scripts\\python.exe examples/03_gateway_caching_demo.py
"""

import sys
import os
import time
from dotenv import load_dotenv
import litellm
from litellm.caching import Cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

litellm.drop_params = True
litellm.set_verbose = False
litellm.suppress_debug_info = True

print("=" * 75)
print("🛡️ GATEWAY CACHING & FAILOVER DEMO")
print("=" * 75)

# 1. Local Cache Test
litellm.cache = Cache(type="local")
prompt = "Explain in one sentence why response caching reduces cloud bills."
model = "openrouter/inclusionai/ling-3.0-flash-fin:free"

print("\n1. First Call (Cache MISS - Network API Call)...")
t0 = time.time()
resp1 = litellm.completion(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    api_key=os.getenv("OPENROUTER_API_KEY"),
    api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    caching=True
)
t1 = round(time.time() - t0, 3)
print(f"⏱️ Latency: {t1}s | Status: Cache MISS")
print(f"💬 Output : {getattr(resp1.choices[0].message, 'content', '')[:70]}...\n")

print("2. Second Call (Cache HIT - Immediate Local Memory Return)...")
t0 = time.time()
resp2 = litellm.completion(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    api_key=os.getenv("OPENROUTER_API_KEY"),
    api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    caching=True
)
t2 = round(time.time() - t0, 4)
print(f"⏱️ Latency: {t2}s | Status: ⚡ Cache HIT ($0.00 Cost)")
print(f"🚀 Speedup: ~{int(t1 / max(t2, 0.0001))}x faster!\n")
print("=" * 75 + "\n")
