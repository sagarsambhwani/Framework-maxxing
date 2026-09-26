"""Unified CLI Entrypoint for the Framework-maxxing Architecture.

Commands:
    1. server    - Launches FastAPI ChatGPT Pro Web UI with SSE streaming & voice mode
    2. agent     - Runs LangGraph Stateful Autonomous Research Agent in terminal mode
    3. marketing - Runs Agentic Multi-Channel Marketing Campaign Generator
    4. ingest    - Runs Enterprise Document Ingestion & Tri-Brid RAG pipeline
    5. benchmark - Runs comparative speed, latency, and TTFT benchmarks
    6. eval      - Runs Enterprise AI Evaluation & Benchmarking Suite
    7. alerts    - Scans latest local trace for production SLA violations
    8. phoenix   - Launches local Arize Phoenix visual tracing UI on http://localhost:6006

Usage:
    .venv\\Scripts\\python.exe main.py server
    .venv\\Scripts\\python.exe main.py agent "Design an AI Gateway" --debug
"""

import sys
import argparse

# Ensure proper Unicode / UTF-8 rendering on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.common.config import settings
from src.common.logging import debug_log
from src.cli import (
    run_server,
    run_agent,
    run_marketing,
    run_ingest,
    run_benchmark,
    run_evaluation,
    run_phoenix_server,
    run_alert_check,
)


def main():
    """Parses command-line arguments and dispatches to appropriate handler."""
    parser = argparse.ArgumentParser(
        description="Framework-maxxing AI Gateway & Autonomous Agent CLI"
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose diagnostic debug logging")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    server_parser = subparsers.add_parser("server", help="Launch FastAPI Web Server on http://localhost:8080")
    server_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    agent_parser = subparsers.add_parser("agent", help="Run LangGraph Autonomous Research Agent")
    agent_parser.add_argument("query", nargs="?", default="Evaluate multi-cloud LLM gateway latency and caching", help="Research query text")
    agent_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    mkt_parser = subparsers.add_parser("marketing", help="Run Agentic Marketing Campaign Generator")
    mkt_parser.add_argument("brief", nargs="?", default="Launch an AI Gateway reducing LLM costs by 70% with 0ms caching", help="Marketing brief")
    mkt_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    subparsers.add_parser("phoenix", help="Launch Arize Phoenix Local Visual Dashboard on http://localhost:6006")
    subparsers.add_parser("alerts", help="Scan latest trace file for SLA and security violations")

    bench_parser = subparsers.add_parser("benchmark", help="Run multi-provider speed & latency benchmark")
    bench_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    eval_parser = subparsers.add_parser("eval", help="Run Enterprise AI Evaluation & Benchmarking Suite")
    eval_parser.add_argument("--export-report", default="evaluation_report.md", help="Path to export Markdown report")
    eval_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    ingest_parser = subparsers.add_parser("ingest", help="Run Enterprise Document Ingestion & Tri-Brid RAG")
    ingest_parser.add_argument("file", nargs="?", default=None, help="Optional document path to ingest")
    ingest_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()

    if getattr(args, "debug", False):
        settings.DEBUG_MODE = True
        debug_log("🔍 [DEBUG:INIT]", "Verbose diagnostic debug logging ENABLED")

    dispatch = {
        "server": run_server,
        "agent": lambda: run_agent(args.query),
        "marketing": lambda: run_marketing(args.brief),
        "phoenix": run_phoenix_server,
        "alerts": run_alert_check,
        "benchmark": run_benchmark,
        "eval": lambda: run_evaluation(export_path=args.export_report),
        "ingest": lambda: run_ingest(file_path=args.file),
    }

    handler = dispatch.get(args.command, run_server)
    handler()


if __name__ == "__main__":
    main()
