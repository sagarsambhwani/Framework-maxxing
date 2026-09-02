"""Unified End-to-End Pipeline Orchestrator."""
import sys
import time
import uuid
from typing import Any, Dict, Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from src.agent.graph import run_research_agent
from src.gateway.router import get_gateway
from src.observability.tracer import get_tracer
from src.observability.metrics import MetricsCollector
from src.guardrails.rails_manager import get_guardrails_manager
from src.common.config import settings

console = Console(highlight=False)


class UnifiedResearchPipeline:
    """End-to-end pipeline connecting OpenRouter routing, LiteLLM Gateway,

    Langfuse Observability, NeMo Guardrails, and LangGraph Agent.
    """

    def __init__(self):
        self.gateway = get_gateway()
        self.tracer = get_tracer()
        self.guardrails = get_guardrails_manager()
        self.metrics = MetricsCollector()

    def execute(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """Execute the end-to-end research query through the protected AI pipeline."""
        session_id = f"pipeline-{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        if verbose:
            console.print(Panel(f"[bold cyan]Input Query:[/bold cyan] {query}\n[dim]Session ID: {session_id}[/dim]", title="🚀 Starting AI Pipeline"))

        # Step 1: Input Guardrail Validation
        if verbose:
            console.print("[bold yellow]1. NeMo Guardrails: Evaluating Input Safety...[/bold yellow]")

        input_check = self.guardrails.validate_input(query)
        if not input_check["allowed"]:
            if verbose:
                console.print(Panel(f"[bold red]❌ BLOCKED BY GUARDRAILS[/bold red]\n{input_check['reason']}", style="red"))
            return {
                "session_id": session_id,
                "status": "BLOCKED",
                "reason": input_check["reason"],
                "report": None,
                "duration_seconds": round(time.time() - start_time, 3)
            }

        if verbose:
            console.print(f"[green]✓ Input Safety Passed[/green] [dim]({input_check['reason']})[/dim]")

        # Step 2: LangGraph Execution (Planner -> Tools -> Synthesizer)
        if verbose:
            console.print("[bold yellow]2. LangGraph Agent: Executing Planner & Tool Caller...[/bold yellow]")

        agent_state = run_research_agent(query=query, session_id=session_id)
        plan = agent_state.get("plan")
        findings = agent_state.get("findings", [])
        final_report = agent_state.get("final_report", "")

        # Display Planning Breakdown
        if verbose and plan:
            table = Table(title="📋 Autonomous Research Plan", show_header=True, header_style="bold magenta")
            table.add_column("Step #", width=8)
            table.add_column("Description", width=45)
            table.add_column("Tool", width=16)
            table.add_column("Status", width=12)

            for step in plan.get("steps", []):
                status_color = "green" if step.get("status") == "completed" else "yellow"
                table.add_row(
                    str(step.get("step_id")),
                    step.get("description"),
                    step.get("tool"),
                    f"[{status_color}]{step.get('status')}[/{status_color}]"
                )
            console.print(table)

        # Step 3: Observability & Metrics
        duration = round(time.time() - start_time, 3)
        self.metrics.record_llm_call(
            model=settings.PRIMARY_MODEL,
            prompt_tokens=len(query.split()) * 5,
            completion_tokens=len(final_report.split()) if final_report else 50,
            latency=duration
        )

        if verbose:
            console.print("\n[bold yellow]3. Synthesized Research Report:[/bold yellow]")
            if final_report:
                console.print(Panel(Markdown(final_report), title="📊 Final Executive Report", border_style="green"))

            metrics_summary = self.metrics.get_summary()
            summary_table = Table(title="📈 Pipeline Telemetry & Observability (Langfuse Integrated)", show_header=True)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="bold green")
            summary_table.add_row("Execution Duration", f"{duration}s")
            summary_table.add_row("Tools Executed", str(len(findings)))
            summary_table.add_row("Estimated Tokens", str(metrics_summary["total_tokens"]))
            summary_table.add_row("Estimated Cost (USD)", f"${metrics_summary['estimated_cost_usd']}")
            summary_table.add_row("Langfuse Tracing Status", "Active" if self.tracer.is_live else "Local Audit Mode")
            console.print(summary_table)

        return {
            "session_id": session_id,
            "status": "SUCCESS",
            "plan": plan,
            "findings_count": len(findings),
            "report": final_report,
            "duration_seconds": duration,
            "metrics": self.metrics.get_summary()
        }


def run_pipeline(query: str, verbose: bool = True) -> Dict[str, Any]:
    pipeline = UnifiedResearchPipeline()
    return pipeline.execute(query=query, verbose=verbose)
