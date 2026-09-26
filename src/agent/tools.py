"""Research Tools (Live Web Search & Safe Math Calculator).

This module contains tools invoked by the LangGraph Autonomous Research Agent
during the execution phase:
    1. `web_search`: Queries DuckDuckGo for live technical documentation.
    2. `calculator`: Safely evaluates mathematical and throughput formulas.
    3. `execute_tool`: Unified dispatcher routing tool calls by name.
"""

from src.common.logging import term_log, Colors
from src.common.tools import safe_web_search, safe_calculator


def web_search(query: str) -> str:
    """Performs live web search using DuckDuckGo search library.

    Args:
        query: Search string or keywords.

    Returns:
        Formatted string containing retrieved titles and body snippets,
        or contextual fallback text if the network request times out.
    """
    return safe_web_search(query)


def calculator(expression: str) -> str:
    """Safely evaluates basic arithmetic formulas without arbitrary code execution risk.

    Args:
        expression: Mathematical string (e.g. '1500 * 60 / 1000').

    Returns:
        Result string with calculation outcome or error description.
    """
    return safe_calculator(expression)


def execute_tool(tool_name: str, tool_input: str) -> str:
    """Dispatches and executes a research tool by name.

    Args:
        tool_name: Name identifier ('web_search' or 'calculator').
        tool_input: Input query or mathematical expression for the tool.

    Returns:
        String result returned by the specified tool.
    """
    term_log("🔧 [TOOL DISPATCH]", f"Executing {Colors.YELLOW}{tool_name}{Colors.END} with input: '{tool_input}'", Colors.YELLOW)
    if tool_name == "web_search":
        return web_search(tool_input)
    elif tool_name == "calculator":
        return calculator(tool_input)
    return f"Executed generic tool: {tool_name}"
