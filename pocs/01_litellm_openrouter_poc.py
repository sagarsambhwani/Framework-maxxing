"""POC 1: OpenRouter Multi-Model Routing & LiteLLM Gateway Fallback Demonstration."""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from src.gateway.router import get_gateway
from src.common.config import settings

console = Console(highlight=False)


def run_litellm_poc():
    console.print(Panel("[bold cyan]POC 1: LiteLLM Gateway with OpenRouter Dynamic Routing[/bold cyan]\n"
                        "Demonstrates multi-model aliases, fallback chains, and latency routing.", title="🧪 POC Suite"))

    gateway = get_gateway()
    test_queries = [
        ("fast-researcher", "What are 3 advantages of LiteLLM in an enterprise architecture?"),
        ("reasoning-planner", "Break down how to optimize LLM latency and throughput into 3 bullet points."),
        ("openrouter-claude", "Explain the role of model fallback mechanisms.")
    ]

    table = Table(title="Gateway Routing & Execution Results", show_header=True, header_style="bold magenta")
    table.add_column("Requested Alias", style="cyan", width=20)
    table.add_column("Resolved Model / Backend", style="green", width=38)
    table.add_column("Latency (s)", style="yellow", width=12)
    table.add_column("Tokens", style="white", width=10)
    table.add_column("Routing Mode", style="blue", width=18)

    for alias, query in test_queries:
        console.print(f"[bold]Querying alias:[/bold] {alias}...")
        response = gateway.completion(
            model=alias,
            messages=[{"role": "user", "content": query}],
            max_tokens=150
        )

        table.add_row(
            alias,
            response.get("model", "unknown")[:35],
            str(response.get("latency_seconds", 0)),
            str(response.get("usage", {}).get("total_tokens", 0)),
            response.get("routing_mode", "direct")
        )

        console.print(f"[dim]Sample Snippet: {response.get('content', '')[:120]}...[/dim]\n")

    console.print(table)
    console.print("[bold green]✓ POC 1 Completed Successfully![/bold green]\n")


if __name__ == "__main__":
    run_litellm_poc()
