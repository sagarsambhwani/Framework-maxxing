"""Integration Tests for LangGraph Stateful Autonomous Research Agent.

Validates:
    1. Tool execution for safe math evaluation and web searching.
    2. End-to-end graph state progression (Guardrail -> Planner -> Executor -> Synthesizer).
"""

import pytest
from src.agent.tools import execute_tool
from src.agent.graph import research_agent


def test_tool_execution():
    """Verifies that individual research tools return expected evaluation outputs."""
    # Test safe calculator arithmetic
    calc_res = execute_tool("calculator", "25 * 4 + 10")
    assert "110" in calc_res

    # Test web search dispatcher
    search_res = execute_tool("web_search", "OpenRouter")
    assert len(search_res) > 0


def test_research_agent_execution():
    """Verifies end-to-end execution of the LangGraph state machine."""
    initial_state = {
        "query": "Research the benefits of Langfuse for AI cost tracking",
        "session_id": "test-agent-session",
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "plan_steps": [],
        "findings": [],
        "final_report": "",
        "iteration_count": 0
    }
    state = research_agent.invoke(initial_state)

    assert state is not None
    assert state.get("guardrail_allowed") is True
    assert state.get("plan_steps") is not None
    assert len(state.get("findings", [])) > 0
    assert state.get("final_report") is not None
    assert len(state.get("final_report")) > 50
