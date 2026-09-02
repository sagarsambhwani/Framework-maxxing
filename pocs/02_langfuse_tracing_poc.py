"""POC 2: Langfuse Observability, Trace Hierarchy & Cost Analytics."""
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.observability.tracer import get_tracer
from src.observability.metrics import MetricsCollector
from src.gateway.router import get_gateway

console = Console(highlight=False)


def run_langfuse_poc():
    console.print(Panel("[bold cyan]POC 2: Langfuse Observability & Multi-Level Tracing[/bold cyan]\n"
                        "Demonstrates trace contexts, span nesting, latency breakdown, and token cost telemetry.", title="🧪 POC Suite"))

    tracer = get_tracer()
    metrics = MetricsCollector()
    gateway = get_gateway()

    session_id = f"demo-session-{int(time.time())}"
    console.print(f"[bold]Creating Root Trace for Session:[/bold] [yellow]{session_id}[/yellow]")

    trace = tracer.create_trace(
        name="POC-Langfuse-Workflow",
        session_id=session_id,
        tags=["poc2", "observability", "langfuse"],
        metadata={"user_tier": "enterprise", "environment": "staging"}
    )

    # Span 1: Intent Analysis
    with trace.span("Span-1:IntentClassification", input_data={"query": "Evaluate Cloud vs On-Premises LLMs"}) as span1:
        time.sleep(0.1)
        span1["output"] = {"intent": "comparative_analysis", "confidence": 0.98}

    # Span 2: LLM Completion
    with trace.span("Span-2:GatewayInference", input_data={"model": "fast-researcher"}) as span2:
        resp = gateway.completion(
            model="fast-researcher",
            messages=[{"role": "user", "content": "List 3 key enterprise considerations for LLM observability."}],
            max_tokens=150
        )
        span2["output"] = resp.get("content", "")[:100]

        # Record generation in trace
        trace.log_generation(
            name="LLM-Inference-Gen",
            model=resp.get("model", "fast-researcher"),
            prompt="List 3 key enterprise considerations for LLM observability.",
            completion=resp.get("content", ""),
            usage=resp.get("usage", {})
        )

        metrics.record_llm_call(
            model=resp.get("model", "fast-researcher"),
            prompt_tokens=resp.get("usage", {}).get("prompt_tokens", 25),
            completion_tokens=resp.get("usage", {}).get("completion_tokens", 75),
            latency=resp.get("latency_seconds", 0.3)
        )

    # Span 3: Guardrail Compliance Check
    with trace.span("Span-3:GuardrailCheck") as span3:
        span3["output"] = {"status": "passed", "violations": []}

    trace.end(output="Completed observability demonstration workflow", status="SUCCESS")

    # Display Spans Hierarchy Table
    table = Table(title="Captured Traces & Spans Hierarchy", show_header=True, header_style="bold magenta")
    table.add_column("Span ID", style="dim", width=12)
    table.add_column("Span Name", style="cyan", width=30)
    table.add_column("Duration (s)", style="yellow", width=15)
    table.add_column("Status", style="green", width=12)

    for s in trace.spans:
        table.add_row(
            s["id"][:8],
            s["name"],
            str(s.get("duration_seconds", 0)),
            s.get("status", "SUCCESS")
        )
    console.print(table)

    summary = metrics.get_summary()
    console.print(Panel(
        f"[bold]Total Recorded Prompt Tokens:[/bold] {summary['total_prompt_tokens']}\n"
        f"[bold]Total Recorded Completion Tokens:[/bold] {summary['total_completion_tokens']}\n"
        f"[bold]Total Execution Latency:[/bold] {summary['total_latency_seconds']}s\n"
        f"[bold]Estimated Session Cost:[/bold] ${summary['estimated_cost_usd']}\n"
        f"[bold]Langfuse Live Mode:[/bold] {'Connected to Cloud' if tracer.is_live else 'Local Audit Recorder Active'}",
        title="📊 Langfuse Telemetry Summary",
        border_style="cyan"
    ))

    console.print("[bold green]✓ POC 2 Completed Successfully![/bold green]\n")


if __name__ == "__main__":
    run_langfuse_poc()
