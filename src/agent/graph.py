"""LangGraph StateGraph Definition for Autonomous Research Agent."""
import uuid
from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, START, END

from src.agent.state import ResearchState
from src.agent.planner import research_planner_node
from src.agent.executor import tool_executor_node
from src.agent.evaluator import evaluator_node, should_continue_research
from src.agent.synthesizer import research_synthesizer_node
from src.guardrails.rails_manager import get_guardrails_manager
from src.observability.tracer import get_tracer


def input_guardrail_node(state: ResearchState) -> Dict[str, Any]:
    """Inspects the query at graph entry using NeMo Guardrails."""
    guardrails = get_guardrails_manager()
    query = state.get("query", "")

    result = guardrails.validate_input(query)

    if not result["allowed"]:
        return {
            "guardrail_status": result,
            "sanitized_query": "",
            "error": result["reason"],
            "final_report": f"### Request Denied by Safety Guardrails\n\n**Reason:** {result['reason']}"
        }

    return {
        "guardrail_status": result,
        "sanitized_query": result.get("sanitized_prompt", query),
        "error": None
    }


def error_handler_node(state: ResearchState) -> Dict[str, Any]:
    """Node executed when safety rails are triggered."""
    return state


def input_guardrail_router(state: ResearchState) -> str:
    """Routes based on input guardrail verdict."""
    guardrail_status = state.get("guardrail_status", {})
    if guardrail_status.get("allowed", True):
        return "planner_node"
    return "error_handler_node"


def build_research_agent_graph():
    """Constructs and compiles the complete LangGraph state machine."""
    workflow = StateGraph(ResearchState)

    # 1. Register Nodes
    workflow.add_node("input_guardrail_node", input_guardrail_node)
    workflow.add_node("planner_node", research_planner_node)
    workflow.add_node("executor_node", tool_executor_node)
    workflow.add_node("evaluator_node", evaluator_node)
    workflow.add_node("synthesizer_node", research_synthesizer_node)
    workflow.add_node("error_handler_node", error_handler_node)

    # 2. Wire Graph Edges
    workflow.add_edge(START, "input_guardrail_node")

    # Guardrail Branching
    workflow.add_conditional_edges(
        "input_guardrail_node",
        input_guardrail_router,
        {
            "planner_node": "planner_node",
            "error_handler_node": "error_handler_node"
        }
    )

    # Planning to Execution
    workflow.add_edge("planner_node", "executor_node")
    workflow.add_edge("executor_node", "evaluator_node")

    # Evaluation Loop / Branching
    workflow.add_conditional_edges(
        "evaluator_node",
        should_continue_research,
        {
            "continue_tools": "executor_node",
            "synthesize": "synthesizer_node",
            "end_with_error": "error_handler_node"
        }
    )

    workflow.add_edge("synthesizer_node", END)
    workflow.add_edge("error_handler_node", END)

    return workflow.compile()


def run_research_agent(query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Executes the research agent graph with Langfuse tracing and returns final state."""
    sess_id = session_id or f"sess-{uuid.uuid4().hex[:8]}"
    tracer = get_tracer()
    trace = tracer.create_trace(
        name="LangGraph:ResearchAgent",
        session_id=sess_id,
        tags=["research-agent", "planner", "tools"],
        metadata={"query": query}
    )

    app = build_research_agent_graph()

    initial_state: ResearchState = {
        "query": query,
        "sanitized_query": "",
        "guardrail_status": {},
        "plan": None,
        "current_step_index": 0,
        "iteration_count": 0,
        "findings": [],
        "needs_replanning": False,
        "final_report": None,
        "error": None,
        "session_id": sess_id,
        "telemetry": {}
    }

    final_state = app.invoke(initial_state)

    trace.end(
        output=final_state.get("final_report", "")[:200],
        status="ERROR" if final_state.get("error") else "SUCCESS"
    )

    return final_state
