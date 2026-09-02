"""POC 4: LangGraph Research Planner & Tool Caller Agent Demonstration."""
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
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from src.agent.graph import run_research_agent

console = Console(highlight=False)


def run_agent_poc():
    console.print(Panel("[bold cyan]POC 4: LangGraph Autonomous Research Planner & Tool Caller[/bold cyan]\n"
                        "Demonstrates multi-node state graph, query decomposition, tool dispatch, and report synthesis.", title="🧪 POC Suite"))

    query = "Analyze the latency overhead and throughput bottlenecks of LiteLLM Gateway in high-concurrency systems"
    console.print(f"[bold yellow]Research Goal:[/bold yellow] {query}\n")

    state = run_research_agent(query=query)

    plan = state.get("plan")
    findings = state.get("findings", [])
    report = state.get("final_report", "")

    if plan:
        table = Table(title="Generated Research Execution Plan", show_header=True, header_style="bold magenta")
        table.add_column("Step #", style="cyan", width=8)
        table.add_column("Goal / Description", style="white", width=48)
        table.add_column("Tool", style="yellow", width=16)
        table.add_column("Status", style="green", width=12)

        for step in plan.get("steps", []):
            table.add_row(
                str(step.get("step_id")),
                step.get("description"),
                step.get("tool"),
                step.get("status")
            )
        console.print(table)

    console.print("\n[bold]Tool Execution Evidence Log:[/bold]")
    for idx, f in enumerate(findings, 1):
        console.print(f"[bold cyan]• Tool Step {idx} ({f.get('tool')}):[/bold cyan] [dim]{f.get('input')}[/dim]")
        console.print(f"  [green]Result:[/green] {str(f.get('result'))[:150]}...\n")

    if report:
        console.print(Panel(Markdown(report), title="📊 Final Synthesized Research Report", border_style="green"))

    console.print("[bold green]✓ POC 4 Completed Successfully![/bold green]\n")


if __name__ == "__main__":
    run_agent_poc()
