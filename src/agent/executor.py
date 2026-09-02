"""Tool Caller / Execution Node: Executes individual research tools according to plan."""
from typing import Any, Dict, List
from src.agent.state import ResearchState
from src.agent.tools import execute_tool_call
from src.observability.tracer import get_tracer


def tool_executor_node(state: ResearchState) -> Dict[str, Any]:
    """LangGraph node that executes the active research step using designated tools."""
    plan = state.get("plan")
    if not plan or not plan.get("steps"):
        return {"needs_replanning": False}

    step_index = state.get("current_step_index", 0)
    steps = plan["steps"]

    if step_index >= len(steps):
        return {"needs_replanning": False}

    current_step = steps[step_index]
    tool_name = current_step.get("tool", "web_search")
    tool_input = current_step.get("tool_input", "")

    tracer = get_tracer()
    trace_record = tracer.active_traces.get(state.get("session_id", ""))

    # Execute tool within traced span
    if trace_record:
        with trace_record.span(f"Tool:{tool_name}", input_data={"tool_input": tool_input}):
            tool_output = execute_tool_call(tool_name, tool_input)
    else:
        tool_output = execute_tool_call(tool_name, tool_input)

    # Update step record
    current_step["status"] = "completed"
    current_step["result"] = tool_output

    findings = list(state.get("findings", []))
    findings.append({
        "step_id": current_step.get("step_id"),
        "step": current_step.get("description"),
        "tool": tool_name,
        "input": tool_input,
        "result": tool_output
    })

    return {
        "plan": plan,
        "findings": findings,
        "current_step_index": step_index + 1
    }
