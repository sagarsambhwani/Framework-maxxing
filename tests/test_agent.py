"""Tests for LangGraph Research Agent."""
import pytest
from src.agent.tools import execute_tool_call
from src.agent.graph import build_research_agent_graph, run_research_agent


def test_tool_execution():
    calc_res = execute_tool_call("calculator", "25 * 4 + 10")
    assert "110" in calc_res

    search_res = execute_tool_call("web_search", "OpenRouter")
    assert len(search_res) > 0


def test_research_agent_graph_build():
    app = build_research_agent_graph()
    assert app is not None


def test_research_agent_execution():
    state = run_research_agent("Research the benefits of Langfuse for AI cost tracking")
    assert state is not None
    assert state.get("plan") is not None
    assert len(state.get("findings", [])) > 0
    assert state.get("final_report") is not None
