"""LangGraph Research Agent State Definition."""
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    query: str
    session_id: str
    guardrail_allowed: bool
    guardrail_reason: str
    plan_steps: List[Dict[str, str]]
    findings: List[Dict[str, str]]
    final_report: str
    iteration_count: int
