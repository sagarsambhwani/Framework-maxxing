"""07_observability_comparison_demo.py - Multi-Target Observability Comparison.

Demonstrates side-by-side observability across 4 backends:
    1. Local JSON Tracing: Inspect `traces/latest_trace.json` with exact TTFT, token counts, and costs.
    2. Arize Phoenix: Local visual evaluation & tracing workbench.
    3. OpenTelemetry (OTel): Standardized distributed spans.
    4. Langfuse Cloud: Cloud telemetry dashboard.
    5. Production Alert Engine: Automated SLA violation detection.

Run with:
    .venv\\Scripts\\python.exe examples/07_observability_comparison_demo.py
"""

import sys
import os
import json
import uuid
import time

# Add root directory to pythonpath
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.observability.tracer import tracer
from src.observability.alerts import ProductionAlertEngine
from src.workflows.marketing.graph import marketing_workflow
from src.common.logging import print_banner, term_log, Colors

if __name__ == "__main__":
    session_id = f"obs-{uuid.uuid4().hex[:6]}"
    brief = "Launch an ultra-fast AI Gateway that reduces inference costs by 70% with 0ms in-memory caching."

    print_banner(
        "MULTI-TARGET OBSERVABILITY COMPARISON DEMO",
        f"Session: {session_id} | Targets: Local JSON, Phoenix, OTel, Langfuse"
    )

    term_log("🚀 [RUNNING WORKFLOW]", "Executing agentic workflow to capture multi-target traces...", Colors.BLUE)
    
    # 1. Execute workflow (automatically dispatches spans to all 4 backends)
    final_state = marketing_workflow.invoke({
        "brief": brief,
        "product_name": "HyperGateway AI",
        "target_audience": "Enterprise CTOs",
        "brand_voice": "Technical & Authoritative",
        "target_channels": ["twitter", "linkedin", "email"],
        "session_id": session_id,
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "research_insights": [],
        "campaign_angles": [],
        "copy_drafts": {},
        "critic_feedback": [],
        "critic_approved": False,
        "revision_count": 0,
        "final_campaign": {}
    })

    # 2. Finalize local trace
    final_trace = tracer.finalize_trace(session_id)

    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}{Colors.GREEN}📊 OBSERVABILITY BACKEND COMPARISON BREAKDOWN{Colors.END}")
    print("=" * 80)

    # A. Local JSON Trace
    print(f"\n1. 📁 {Colors.CYAN}LOCAL JSON TRACING (100% Offline & Transparent){Colors.END}")
    print(f"   • File Location        : traces/latest_trace.json (and traces/{session_id}.json)")
    print(f"   • Total Spans Captured : {len(final_trace.get('spans', []))} spans")
    print(f"   • Total Tokens Tracked : {final_trace.get('total_tokens')} tokens ({final_trace.get('total_prompt_tokens')} in / {final_trace.get('total_completion_tokens')} out)")
    print(f"   • Estimated Cost       : ${final_trace.get('estimated_cost_usd', 0.0):.6f}")

    # B. Arize Phoenix
    print(f"\n2. 🦅 {Colors.YELLOW}ARIZE PHOENIX (Local Visual Workbench){Colors.END}")
    print(f"   • Local Dashboard URL  : http://localhost:6006")
    print(f"   • How to Launch UI     : Run '.venv\\Scripts\\python.exe main.py phoenix' in another terminal")
    print(f"   • Capabilities         : Visual execution graphs, embedding drift analysis, RAG evaluations")

    # C. OpenTelemetry (OTel)
    print(f"\n3. 🔭 {Colors.MAGENTA}OPENTELEMETRY STANDARD (OTel Spans){Colors.END}")
    print(f"   • Format               : W3C OpenTelemetry Span Standard")
    print(f"   • Target APMs          : Direct export to Datadog, New Relic, Dynatrace, Jaeger")

    # D. Langfuse Cloud
    print(f"\n4. 📈 {Colors.BLUE}LANGFUSE CLOUD (Remote SaaS Dashboard){Colors.END}")
    print(f"   • Cloud Dashboard URL  : https://cloud.langfuse.com")
    print(f"   • Status               : Synced in background")

    # 3. Run Production Alert Engine on the Captured Trace
    print("\n" + "-" * 80)
    term_log("🔍 [RUNNING ALERT ENGINE]", "Scanning trace metrics against production SLAs...", Colors.BLUE)
    alerts = ProductionAlertEngine.evaluate_trace(final_trace)
    ProductionAlertEngine.render_alerts(alerts)

    print("=" * 80 + "\n")
