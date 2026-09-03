"""Unified CLI Entrypoint for the Framework-maxxing Architecture.

This module provides a single, polished command-line interface for running:
    1. `python main.py server [--debug]`
       Launches the FastAPI ChatGPT Pro Web UI with Server-Sent Events (SSE) streaming,
       Groq Whisper Turbo voice mode, and live colored terminal logging on port 8080.

    2. `python main.py agent "your query" [--debug]`
       Runs the LangGraph Stateful Autonomous Research Agent in terminal mode with
       automated safety evaluation, planning, tool dispatching, and report synthesis.

    3. `python main.py benchmark [--debug]`
       Executes comparative speed, latency, and TTFT benchmarks across Groq LPUs,
       Google Gemini, and OpenRouter endpoints.

    4. `python main.py eval [--export-report path.md]`
       Runs the Enterprise AI Evaluation & Benchmarking Suite (Safety, RAG Triad, Tools, Speed).

Usage Examples:
    .venv\\Scripts\\python.exe main.py server
    .venv\\Scripts\\python.exe main.py eval
    .venv\\Scripts\\python.exe main.py agent "Design an AI Gateway with caching" --debug
"""

import sys
import os
import argparse
import uuid
import time
import uvicorn

# Ensure proper Unicode / UTF-8 rendering on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.common.config import settings
from src.common.logging import term_log, debug_log, print_banner, Colors


def run_server():
    """Starts the FastAPI Web Application & ChatGPT dark-themed interface."""
    from src.server.app import create_app
    app = create_app()

    print_banner(
        "CHATGPT PRO FASTAPI + JAVASCRIPT SERVER",
        f"Web UI: http://localhost:{settings.SERVER_PORT} | Terminal Logs: ACTIVE | Debug Mode: {'ON' if settings.DEBUG_MODE else 'OFF'}"
    )
    uvicorn.run(app, host="127.0.0.1", port=settings.SERVER_PORT, log_level="warning")


def run_agent(query: str):
    """Executes the LangGraph Autonomous Research Agent workflow from the terminal.

    Args:
        query: Research question or instruction to investigate.
    """
    from src.agent.graph import research_agent
    session_id = f"agent-{uuid.uuid4().hex[:6]}"

    print_banner("LANGGRAPH AUTONOMOUS RESEARCH AGENT", f"Query: '{query}' | Session: {session_id} | Debug Mode: {'ON' if settings.DEBUG_MODE else 'OFF'}")
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
        for idx, step in enumerate(final_state["plan_steps"], 1):
            print(f"     Step {idx}: [{step['tool']}] -> {step['input']}")
        print(f"\n🔍 Tools Executed  : {len(final_state['findings'])} tools completed.")
        print(f"\n📝 Final Report    :\n\n{final_state['final_report']}")
        print(f"\n⏱️  Total Duration   : {dur}s | Traces synced to Langfuse Cloud")
    else:
        print(f"🛡️  NeMo Guardrails: {Colors.RED}BLOCKED{Colors.END}")
        print(f"     Reason: {final_state['guardrail_reason']}")
    print("=" * 80 + "\n")


def run_benchmark():
    """Executes comparative latency benchmarks across all supported cloud providers."""
    from src.gateway.router import gateway
    print_banner("MULTI-PROVIDER SPEED & LATENCY BENCHMARK", f"Testing Groq LPU vs Google Gemini vs OpenRouter | Debug Mode: {'ON' if settings.DEBUG_MODE else 'OFF'}")

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


def run_evaluation(export_path: str = "evaluation_report.md"):
    """Executes the enterprise evaluation & benchmarking suite."""
    from src.evaluation.runner import run_evaluation_suite
    run_evaluation_suite(export_path=export_path)


def run_marketing(brief: str):
    """Executes the Agentic Marketing Campaign Workflow."""
    from src.workflows.marketing.graph import marketing_workflow
    session_id = f"mkt-{uuid.uuid4().hex[:6]}"

    print_banner("AGENTIC MARKETING WORKFLOW", f"Brief: '{brief[:60]}...' | Session: {session_id}")
    start_t = time.time()

    final_state = marketing_workflow.invoke({
        "brief": brief,
        "product_name": "New AI Product",
        "target_audience": "Tech Leaders & Engineers",
        "brand_voice": "Authoritative, Direct & Engaging",
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
    dur = round(time.time() - start_t, 2)

    campaign = final_state.get("final_campaign", {})
    assets = campaign.get("assets", {})

    print("\n" + "=" * 80)
    print("📣 APPROVED MULTI-CHANNEL CAMPAIGN DELIVERABLES")
    print("=" * 80)
    print(f"Status           : {Colors.GREEN}{campaign.get('status', 'APPROVED')}{Colors.END}")
    print(f"Reflection Loops : {campaign.get('revisions_executed', 1)} iteration(s)")
    print("-" * 80)

    # Twitter
    tw = assets.get("twitter_x", {})
    print(f"\n🐦 [TWITTER / X POST] ({tw.get('metrics', {}).get('char_count', 0)}/280 chars):")
    print(f"{tw.get('copy')}\n")

    # LinkedIn
    li = assets.get("linkedin", {})
    print(f"💼 [LINKEDIN THOUGHT LEADERSHIP] ({li.get('metrics', {}).get('word_count', 0)} words):")
    print(f"{li.get('copy')}\n")

    # Email
    em = assets.get("email", {})
    print(f"📧 [EMAIL NURTURE]:")
    print(f"{em.get('copy')}\n")

    print("=" * 80)
    print(f"⏱️  Campaign created and verified in {dur}s | Traces synced to Langfuse Cloud")
    print("=" * 80 + "\n")


def main():
    """Parses command-line arguments and dispatches to appropriate handler."""
    parser = argparse.ArgumentParser(
        description="Framework-maxxing AI Gateway & Autonomous Agent CLI"
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose diagnostic debug logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # 1. 'server' subcommand
    server_parser = subparsers.add_parser("server", help="Launch FastAPI Web Server on http://localhost:8080")
    server_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    # 2. 'agent' subcommand
    agent_parser = subparsers.add_parser("agent", help="Run LangGraph Autonomous Research Agent")
    agent_parser.add_argument(
        "query",
        nargs="?",
        default="Evaluate multi-cloud LLM gateway latency and caching",
        help="Research query text"
    )
    agent_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    # 3. 'marketing' subcommand
    mkt_parser = subparsers.add_parser("marketing", help="Run Agentic Marketing Campaign Generator")
    mkt_parser.add_argument(
        "brief",
        nargs="?",
        default="Launch an AI Gateway reducing LLM costs by 70% with 0ms caching",
        help="Product marketing brief or campaign objective"
    )
    mkt_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    # 4. 'benchmark' subcommand
    bench_parser = subparsers.add_parser("benchmark", help="Run multi-provider speed & latency benchmark")
    bench_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    # 5. 'eval' subcommand
    eval_parser = subparsers.add_parser("eval", help="Run Enterprise AI Evaluation & Benchmarking Suite")
    eval_parser.add_argument("--export-report", default="evaluation_report.md", help="Path to export Markdown report")
    eval_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()

    if getattr(args, "debug", False):
        settings.DEBUG_MODE = True
        debug_log("🔍 [DEBUG:INIT]", "Verbose diagnostic debug logging ENABLED")

    if args.command == "server":
        run_server()
    elif args.command == "agent":
        run_agent(args.query)
    elif args.command == "marketing":
        run_marketing(args.brief)
    elif args.command == "benchmark":
        run_benchmark()
    elif args.command == "eval":
        run_evaluation(export_path=args.export_report)
    else:
        # Default behavior: start server
        run_server()


if __name__ == "__main__":
    main()
