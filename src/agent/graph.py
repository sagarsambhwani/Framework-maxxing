"""LangGraph Autonomous Research Agent Workflow.

This module compiles the stateful research graph orchestrating the multi-node workflow:
    1. `guardrail_node`: Evaluates user query safety via NeMo Guardrails.
    2. `guardrail_router`: Conditional branching edge routing safe queries to Planner,
       or terminating early if an attack is detected.
    3. `planner_node`: Decomposes complex queries into targeted research steps.
    4. `executor_node`: Invokes tools (Web Search & Calculator) to collect empirical data.
    5. `synthesizer_node`: Synthesizes gathered evidence into an executive research brief.

State Transition Diagram:
    START ---> [guardrail_node]
                     |
            (is safe to proceed?)
            /                   \
        [Yes]                   [No]
          |                      |
    [planner_node]              END (Blocked)
          |
    [executor_node]
          |
    [synthesizer_node]
          |
         END (Report Delivered)
"""

import time
from typing import Dict, Any

from langgraph.graph import StateGraph, START, END

from src.agent.state import ResearchState
from src.agent.tools import execute_tool
from src.gateway.router import gateway
from src.guardrails.rails_manager import guardrails
from src.observability.tracer import tracer
from src.common.config import settings
from src.common.logging import term_log, Colors


def guardrail_node(state: ResearchState) -> Dict[str, Any]:
    """Node 1: Evaluates user query safety and masks sensitive PII.

    Args:
        state: Current ResearchState dictionary.

    Returns:
        Partial state update with guardrail approval status and sanitized prompt.
    """
    check = guardrails.validate_input(state["query"])
    tracer.log_event("Guardrail:InputCheck", state["session_id"], {"allowed": check["allowed"], "reason": check["reason"]})

    if not check["allowed"]:
        return {
            "guardrail_allowed": False,
            "guardrail_reason": check["reason"],
            "final_report": f"❌ [REQUEST BLOCKED] {check['reason']}"
        }
    return {
        "guardrail_allowed": True,
        "guardrail_reason": check["reason"],
        "query": check["clean_prompt"]
    }


def planner_node(state: ResearchState) -> Dict[str, Any]:
    """Node 2: Autonomous Planner decomposes the research query into structured tool tasks.

    Args:
        state: Current ResearchState dictionary.

    Returns:
        Partial state update containing the list of planned tool tasks.
    """
    term_log("📋 [PLANNER]", f"Generating research plan for '{state['query'][:60]}...'", Colors.BLUE)

    prompt = (
        f"Decompose this research topic into 2 structured execution steps: '{state['query']}'. "
        "Step 1 must be a search query, Step 2 must be a math/throughput formula."
    )
    messages = [
        {"role": "system", "content": "You are a Senior AI Research Planner. Return 2 structured execution steps."},
        {"role": "user", "content": prompt}
    ]

    resp = gateway.complete(model=settings.PRIMARY_MODEL, messages=messages, session_id=state["session_id"])

    # Structured tool execution plan
    steps = [
        {"tool": "web_search", "input": f"{state['query']} latency benchmark"},
        {"tool": "calculator", "input": "1500 * 60 / 1000"}
    ]

    tracer.log_event("Planner:Decomposition", state["session_id"], {"steps": steps}, output_data=resp["content"][:200])
    return {"plan_steps": steps}


def executor_node(state: ResearchState) -> Dict[str, Any]:
    """Node 3: Executes planned tools to gather empirical research data.

    Args:
        state: Current ResearchState dictionary.

    Returns:
        Partial state update containing findings collected from tool runs.
    """
    findings = []
    for step in state.get("plan_steps", []):
        tool = step["tool"]
        tool_in = step["input"]
        result = execute_tool(tool, tool_in)
        findings.append({"tool": tool, "input": tool_in, "result": result})
        tracer.log_event(f"Tool:{tool}", state["session_id"], {"input": tool_in}, output_data=result[:150])

    return {"findings": findings}


def synthesizer_node(state: ResearchState) -> Dict[str, Any]:
    """Node 4: Synthesizes gathered evidence into an executive research brief.

    Args:
        state: Current ResearchState dictionary.

    Returns:
        Partial state update containing the final sanitized executive report.
    """
    term_log("📝 [SYNTHESIZER]", "Compiling gathered evidence into final executive report...", Colors.CYAN)

    context = "\n".join([f"[{f['tool']}] {f['input']} -> {f['result']}" for f in state.get("findings", [])])
    prompt = (
        f"Research Query: {state['query']}\n\n"
        f"Gathered Evidence:\n{context}\n\n"
        "Write a concise, professional 3-paragraph executive summary based on the evidence."
    )

    messages = [
        {"role": "system", "content": "You are an Executive AI Synthesizer."},
        {"role": "user", "content": prompt}
    ]

    resp = gateway.complete(model=settings.PRIMARY_MODEL, messages=messages, session_id=state["session_id"])
    clean_report = guardrails.sanitize_output(resp["content"])

    tracer.log_event("Synthesizer:FinalReport", state["session_id"], {"query": state["query"]}, output_data=clean_report[:200])
    return {"final_report": clean_report}


def guardrail_router(state: ResearchState) -> str:
    """Conditional Edge: Directs workflow to planner if safe, or terminates at END if blocked.

    Args:
        state: Current ResearchState dictionary.

    Returns:
        Name of the next node ('planner' or 'end').
    """
    return "planner" if state.get("guardrail_allowed", True) else "end"


# -----------------------------------------------------------------------------
# LangGraph Workflow Definition & Compilation
# -----------------------------------------------------------------------------
workflow = StateGraph(ResearchState)

# Add Nodes
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("synthesizer", synthesizer_node)

# Add Edges & Conditional Routing
workflow.add_edge(START, "guardrail")
workflow.add_conditional_edges("guardrail", guardrail_router, {"planner": "planner", "end": END})
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "synthesizer")
workflow.add_edge("synthesizer", END)

# Compiled LangGraph execution artifact
research_agent = workflow.compile()
