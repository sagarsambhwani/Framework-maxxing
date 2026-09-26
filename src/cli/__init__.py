"""CLI Command Handlers Facade Package.

Exposes high-level execution routines for all CLI subcommands:
- run_server: Fast API web application with chat interface
- run_agent: LangGraph autonomous research agent
- run_marketing: Multi-agent marketing campaign generation
- run_ingest: Enterprise document ingestion & benchmark
- run_benchmark: LLM provider performance & TTFT benchmarking
- run_evaluation: Enterprise evaluation and safety suite
- run_phoenix_server: Arize Phoenix local visual tracing UI
- run_alert_check: Production SLA alerts scanner
"""

from src.cli.server_cmd import run_server
from src.cli.agent_cmd import run_agent
from src.cli.marketing_cmd import run_marketing
from src.cli.ingest_cmd import run_ingest
from src.cli.benchmark_cmd import run_benchmark
from src.cli.eval_cmd import run_evaluation
from src.cli.alerts_cmd import run_phoenix_server, run_alert_check

__all__ = [
    "run_server",
    "run_agent",
    "run_marketing",
    "run_ingest",
    "run_benchmark",
    "run_evaluation",
    "run_phoenix_server",
    "run_alert_check",
]
