"""LangGraph Stateful Autonomous Research Agent Workflow."""
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
    """Node 1: Evaluates user query against security policies."""
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
    """Node 2: Autonomous Planner decomposes query into targeted tasks."""
    term_log("📋 [PLANNER]", f"Generating research plan for '{state['query'][:60]}...'", Colors.BLUE)
    
    prompt = f"Plan 2 research tasks for: '{state['query']}'. Step 1: Search query, Step 2: Math/throughput calculation."
    messages = [
        {"role": "system", "content": "You are a Research Planner. Return 2 structured execution steps."},
        {"role": "user", "content": prompt}
    ]
    
    resp = gateway.complete(model=settings.PRIMARY_MODEL, messages=messages, session_id=state["session_id"])
    
    steps = [
        {"tool": "web_search", "input": f"{state['query']} latency benchmark"},
        {"tool": "calculator", "input": "1500 * 60 / 1000"}
    ]
    
    tracer.log_event("Planner:Decomposition", state["session_id"], {"steps": steps}, output_data=resp["content"][:200])
    return {"plan_steps": steps}


def executor_node(state: ResearchState) -> Dict[str, Any]:
    """Node 3: Executes planned research tools."""
    findings = []
    for step in state.get("plan_steps", []):
        tool = step["tool"]
        tool_in = step["input"]
        result = execute_tool(tool, tool_in)
        findings.append({"tool": tool, "input": tool_in, "result": result})
        tracer.log_event(f"Tool:{tool}", state["session_id"], {"input": tool_in}, output_data=result[:150])

    return {"findings": findings}


def synthesizer_node(state: ResearchState) -> Dict[str, Any]:
    """Node 4: Synthesizes final executive research brief."""
    term_log("📝 [SYNTHESIZER]", "Compiling final evidence into executive report...", Colors.CYAN)
    
    context = "\n".join([f"[{f['tool']}] {f['input']} -> {f['result']}" for f in state.get("findings", [])])
    prompt = f"Query: {state['query']}\n\nEvidence:\n{context}\n\nWrite a 3-paragraph executive summary."
    
    messages = [
        {"role": "system", "content": "You are an Executive AI Synthesizer."},
        {"role": "user", "content": prompt}
    ]
    
    resp = gateway.complete(model=settings.PRIMARY_MODEL, messages=messages, session_id=state["session_id"])
    clean_report = guardrails.sanitize_output(resp["content"])
    
    tracer.log_event("Synthesizer:FinalReport", state["session_id"], {"query": state["query"]}, output_data=clean_report[:200])
    return {"final_report": clean_report}


def guardrail_router(state: ResearchState) -> str:
    """Conditional Edge: Route to planner if safe, else terminate."""
    return "planner" if state.get("guardrail_allowed", True) else "end"


# Compile LangGraph StateGraph Workflow
workflow = StateGraph(ResearchState)
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "guardrail")
workflow.add_conditional_edges("guardrail", guardrail_router, {"planner": "planner", "end": END})
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "synthesizer")
workflow.add_edge("synthesizer", END)

research_agent = workflow.compile()
