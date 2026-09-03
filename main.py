"""Unified CLI Entrypoint for the Framework-maxxing Architecture.

Usage:
    python main.py server               # Launch FastAPI ChatGPT Web UI on http://localhost:8080
    python main.py agent "your query"  # Run LangGraph Autonomous Research Agent
    python main.py benchmark           # Run Multi-Provider Speed & Latency Benchmark
"""

import sys
import os
import argparse
import uuid
import time
import uvicorn

# Ensure UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.common.config import settings
from src.common.logging import term_log, print_banner, Colors


def run_server():
    """Launches the FastAPI Web Server and ChatGPT Interface."""
    from src.server.app import create_app
    app = create_app()

    print_banner(
        "CHATGPT PRO FASTAPI + JAVASCRIPT SERVER",
        f"Web UI: http://localhost:{settings.SERVER_PORT} | Terminal Logs: ACTIVE"
    )
    uvicorn.run(app, host="127.0.0.1", port=settings.SERVER_PORT, log_level="warning")


def run_agent(query: str):
    """Executes the LangGraph Autonomous Research Agent."""
    from src.agent.graph import research_agent
    session_id = f"agent-{uuid.uuid4().hex[:6]}"

    print_banner("LANGGRAPH AUTONOMOUS RESEARCH AGENT", f"Query: '{query}' | Session: {session_id}")
    start_t = time.time()

    initial_state = {
        "query": query,
        "session_id": session_id,
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "plan_steps": [],
        "findings": [],
        "final_report": "",
        "iteration_count": 0
    }

    final_state = research_agent.invoke(initial_state)
    dur = round(time.time() - start_t, 2)

    print("\n" + "=" * 80)
    print("📊 AGENT EXECUTION SUMMARY")
    print("=" * 80)
    if final_state["guardrail_allowed"]:
        print(f"🛡️  NeMo Guardrails: {Colors.GREEN}PASSED{Colors.END}")
        print("📋 Planned Tasks   :")
        for idx, s in enumerate(final_state["plan_steps"], 1):
            print(f"     Step {idx}: [{s['tool']}] -> {s['input']}")
        print(f"\n🔍 Tools Executed  : {len(final_state['findings'])} tools completed.")
        print(f"\n📝 Final Report    :\n\n{final_state['final_report']}")
        print(f"\n⏱️  Total Duration   : {dur}s | Traces synced to Langfuse Cloud")
    else:
        print(f"🛡️  NeMo Guardrails: {Colors.RED}BLOCKED{Colors.END}")
        print(f"     Reason: {final_state['guardrail_reason']}")
    print("=" * 80 + "\n")


def run_benchmark():
    """Runs latency benchmarks across Groq, Google Gemini, and OpenRouter."""
    from src.gateway.router import gateway
    print_banner("MULTI-PROVIDER SPEED & LATENCY BENCHMARK", "Testing Groq LPU vs Google Gemini vs OpenRouter")

    benchmarks = [
        ("groq/qwen/qwen3.8-27b", "⚡ Groq LPU (Qwen 3.8 27B)"),
        ("groq/groq/compound", "🧠 Groq Compound Reasoning"),
        ("gemini/gemma-4-31b-it", "🔵 Google Gemini (Gemma 31B)"),
        ("openrouter/inclusionai/ling-3.0-flash-fin:free", "🟢 OpenRouter (Ling 3.0 Flash)")
    ]

    for model_slug, label in benchmarks:
        t0 = time.time()
        res = gateway.complete(
            model=model_slug,
            messages=[{"role": "user", "content": "Explain token latency in 1 sentence."}],
            max_tokens=40
        )
        print(f"• {label:<35} : {res['latency_s']}s (Tokens: {res['tokens']})")
        print(f"  -> '{res['content'][:70]}...'\n")

    print("✓ Benchmark complete!")


def main():
    parser = argparse.ArgumentParser(description="Framework-maxxing AI Architecture CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Server command
    subparsers.add_parser("server", help="Launch FastAPI Web Server")

    # Agent command
    agent_parser = subparsers.add_parser("agent", help="Run LangGraph Research Agent")
    agent_parser.add_argument("query", nargs="?", default="Evaluate multi-cloud LLM gateway latency and caching", help="Research question")

    # Benchmark command
    subparsers.add_parser("benchmark", help="Run multi-provider speed benchmark")

    args = parser.parse_args()

    if args.command == "server":
        run_server()
    elif args.command == "agent":
        run_agent(args.query)
    elif args.command == "benchmark":
        run_benchmark()
    else:
        # Default behavior if no argument: launch server
        run_server()


if __name__ == "__main__":
    main()
