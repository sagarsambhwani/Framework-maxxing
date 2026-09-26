"""Agent CLI Command Handler."""

import time
import uuid
from src.common.config import settings
from src.common.logging import print_banner, Colors


def run_agent(query: str):
    """Executes the LangGraph Autonomous Research Agent workflow from the terminal."""
    from src.agent.graph import research_agent

    session_id = f"agent-{uuid.uuid4().hex[:6]}"

    print_banner(
        "LANGGRAPH AUTONOMOUS RESEARCH AGENT",
        f"Query: '{query}' | Session: {session_id} | Debug Mode: {'ON' if settings.DEBUG_MODE else 'OFF'}"
    )
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
    if final_state.get("guardrail_allowed", True):
        print(f"🛡️  NeMo Guardrails: {Colors.GREEN}PASSED{Colors.END}")
        print("📋 Planned Tasks   :")
        for idx, step in enumerate(final_state.get("plan_steps", []), 1):
            print(f"     Step {idx}: [{step['tool']}] -> {step['input']}")
        print(f"\n🔍 Tools Executed  : {len(final_state.get('findings', []))} tools completed.")
        print(f"\n📝 Final Report    :\n\n{final_state.get('final_report', '')}")
        print(f"\n⏱️  Total Duration   : {dur}s | Traces synced to Langfuse Cloud")
    else:
        print(f"🛡️  NeMo Guardrails: {Colors.RED}BLOCKED{Colors.END}")
        print(f"     Reason: {final_state.get('guardrail_reason')}")
    print("=" * 80 + "\n")
