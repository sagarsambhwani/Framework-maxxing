"""State definition for LangGraph Research Agent."""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ResearchStep(TypedDict):
    step_id: int
    description: str
    tool: str
    tool_input: str
    status: str  # "pending", "completed", "failed"
    result: Optional[str]


class ResearchPlan(TypedDict):
    thought: str
    steps: List[ResearchStep]


class ResearchState(TypedDict):
    # User Input & Safety
    query: str
    sanitized_query: str
    guardrail_status: Dict[str, Any]

    # Planning & Decomposition
    plan: Optional[ResearchPlan]
    current_step_index: int
    iteration_count: int

    # Collected Tool Findings & Intermediate Context
    findings: List[Dict[str, Any]]
    needs_replanning: bool

    # Final Output & Evaluation
    final_report: Optional[str]
    error: Optional[str]

    # Observability & Session Metadata
    session_id: str
    telemetry: Dict[str, Any]
