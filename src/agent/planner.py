"""Research Planner Node: Breaks user inquiries into structured research steps."""
import json
import re
from typing import Any, Dict
from src.agent.state import ResearchState, ResearchPlan
from src.gateway.router import get_gateway
from src.common.config import settings
from src.common.mock_provider import generate_mock_plan


PLANNER_SYSTEM_PROMPT = """You are an Autonomous Research Planner.
Your job is to break down a complex user research question into 2 to 4 precise, structured steps.
Available Tools:
- web_search: Searches current web info, technology docs, and specifications.
- calculator: Performs math/statistical calculations.
- summarizer: Condenses long text into key insights.
- benchmark_data: Looks up performance, latency, and throughput metrics for AI infrastructure.

Return ONLY a valid JSON object strictly matching this schema:
{
  "thought": "Your reasoning about how to approach the research",
  "steps": [
    {
      "step_id": 1,
      "description": "Short explanation of the step",
      "tool": "tool_name",
      "tool_input": "exact query or argument for the tool"
    }
  ]
}
"""


def research_planner_node(state: ResearchState) -> Dict[str, Any]:
    """LangGraph node that formulates a step-by-step research execution plan."""
    query = state.get("sanitized_query") or state.get("query", "")
    gateway = get_gateway()

    prompt = f"User Research Query: {query}\n\nGenerate the structured execution plan in JSON."
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    plan_data: Dict[str, Any] = {}
    try:
        response = gateway.completion(
            model=settings.PLANNER_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=800,
            metadata={"node": "planner", "session_id": state.get("session_id", "")}
        )
        content = response["content"].strip()

        # Extract JSON from code fence if present
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            raw_json = json_match.group(1)
        else:
            raw_json = content

        plan_data = json.loads(raw_json)
    except Exception as e:
        # Fallback to deterministic plan generator
        plan_data = json.loads(generate_mock_plan(query))

    # Format steps with status
    steps = []
    for s in plan_data.get("steps", []):
        steps.append({
            "step_id": s.get("step_id", len(steps) + 1),
            "description": s.get("description", ""),
            "tool": s.get("tool", "web_search"),
            "tool_input": str(s.get("tool_input", "")),
            "status": "pending",
            "result": None
        })

    structured_plan: ResearchPlan = {
        "thought": plan_data.get("thought", "Decomposed research query."),
        "steps": steps
    }

    return {
        "plan": structured_plan,
        "current_step_index": 0,
        "needs_replanning": False
    }
