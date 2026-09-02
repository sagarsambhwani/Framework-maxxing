from typing import Any, Dict
from src.agent.state import ResearchState
from src.common.config import settings


def should_continue_research(state: ResearchState) -> str:
    """Conditional edge router determining the next step in the LangGraph workflow."""
    # If input guardrail failed, abort directly to end
    guardrail_status = state.get("guardrail_status", {})
    if not guardrail_status.get("allowed", True):
        return "end_with_error"

    plan = state.get("plan")
    if not plan or not plan.get("steps"):
        return "synthesize"

    step_index = state.get("current_step_index", 0)
    total_steps = len(plan["steps"])
    iteration = state.get("iteration_count", 0)

    # If more tool steps remain and we haven't hit iteration limits
    if step_index < total_steps and iteration < settings.MAX_RESEARCH_STEPS:
        return "continue_tools"

    return "synthesize"


def evaluator_node(state: ResearchState) -> Dict[str, Any]:
    """LangGraph node to update loop counts and evaluate state."""
    iteration = state.get("iteration_count", 0) + 1
    return {"iteration_count": iteration}
