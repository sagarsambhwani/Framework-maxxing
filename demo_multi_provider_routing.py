"""demo_multi_provider_routing.py - Multi-Provider Routing & 503 Outage Failover Demo.

Demonstrates:
1. What happens during a real-world provider outage (e.g., Google 503 Service Unavailable)
2. How the Cross-Provider Router (Google Gemini <-> OpenRouter) automatically catches the 503 and saves the request
"""

import os
import sys
import time
from dotenv import load_dotenv
import litellm
from litellm import Router

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

os.environ["OPENROUTER_API_KEY"] = OPENROUTER_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_KEY

litellm.drop_params = True
litellm.set_verbose = False


def print_header(title: str):
    print("\n" + "=" * 75)
    print(f"👉 {title}")
    print("=" * 75)


def run_demo():
    print("=== Multi-Provider Architecture (Google Gemini + OpenRouter) ===")

    # 1. DIRECT CALL TO GOOGLE GEMINI (Showing direct provider behavior)
    print_header("1. Direct Call to Google Gemini (using GEMINI_API_KEY)")
    t0 = time.time()
    try:
        resp_gemini = litellm.completion(
            model="gemini/gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "In 1 concise sentence, what is Google Gemini?"}],
            api_key=GEMINI_KEY
        )
        t_gemini = round(time.time() - t0, 2)
        print(f"⏱️ Latency: {t_gemini}s | Provider: Google AI Studio Direct")
        print(f"💬 Response: {resp_gemini.choices[0].message.content.strip()}\n")
    except Exception as e:
        print(f"⚠️ Direct call encountered provider issue (e.g. 503 High Demand / Outage):")
        print(f"   {e.__class__.__name__}: {str(e)[:120]}...\n")

    # 2. DIRECT CALL TO OPENROUTER
    print_header("2. Direct Call to OpenRouter (using OPENROUTER_API_KEY)")
    t0 = time.time()
    try:
        resp_openrouter = litellm.completion(
            model="openrouter/inclusionai/ling-3.0-flash-fin:free",
            messages=[{"role": "user", "content": "In 1 concise sentence, what is OpenRouter?"}],
            api_key=OPENROUTER_KEY,
            api_base="https://openrouter.ai/api/v1"
        )
        t_or = round(time.time() - t0, 2)
        print(f"⏱️ Latency: {t_or}s | Provider: OpenRouter Broker")
        print(f"💬 Response: {resp_openrouter.choices[0].message.content.strip()}\n")
    except Exception as e:
        print(f"OpenRouter Error: {e}\n")

    # 3. CROSS-PROVIDER AUTOMATED FAILOVER ROUTER
    print_header("3. Cross-Provider Failover Router (Google Gemini -> OpenRouter)")
    print("Scenario: Primary model is Google Gemini. If Google returns 503 High Demand or 429,")
    print("LiteLLM's Router catches it and transparently reroutes to OpenRouter in real-time!\n")

    router = Router(
        model_list=[
            {
                "model_name": "primary-route",
                "litellm_params": {
                    "model": "gemini/gemini-3.6-flash",
                    "api_key": GEMINI_KEY
                }
            },
            {
                "model_name": "fallback-route",
                "litellm_params": {
                    "model": "openrouter/inclusionai/ling-3.0-flash-fin:free",
                    "api_key": OPENROUTER_KEY,
                    "api_base": "https://openrouter.ai/api/v1"
                }
            }
        ],
        fallbacks=[{"primary-route": ["fallback-route"]}]
    )

    t0 = time.time()
    resp_router = router.completion(
        model="primary-route",
        messages=[{"role": "user", "content": "Give 2 key benefits of multi-provider redundancy in production."}]
    )
    t_route = round(time.time() - t0, 2)

    print(f"✓ Zero-Downtime Response received in {t_route}s!")
    print(f"Model that served the request: {resp_router.model}")
    print(f"Response:\n{resp_router.choices[0].message.content.strip()}\n")


if __name__ == "__main__":
    run_demo()
