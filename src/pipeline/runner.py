"""Unified End-to-End Pipeline Orchestrator."""
import sys
import time
import uuid
from typing import Any, Dict

from src.agent.graph import research_agent
from src.guardrails.rails_manager import guardrails
from src.observability.tracer import tracer
from src.common.config import settings
from src.common.logging import term_log, Colors


class UnifiedResearchPipeline:
    """End-to-end pipeline connecting Multi-Provider routing, Langfuse, NeMo Guardrails, and LangGraph."""

    def execute(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        session_id = f"pipe-{uuid.uuid4().hex[:6]}"
        start_time = time.time()

        initial_state = {
            "query": query,
            "session_id": session_id,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "plan_steps": [],
            "findings": [],
            "final_report": "",
            "iteration_count": 0
        }

        final_state = research_agent.invoke(initial_state)
        dur = round(time.time() - start_time, 3)

        if not final_state["guardrail_allowed"]:
            return {
                "session_id": session_id,
                "status": "BLOCKED",
                "reason": final_state["guardrail_reason"],
                "report": None,
                "duration_seconds": dur
            }

        return {
            "session_id": session_id,
            "status": "SUCCESS",
            "plan_steps": final_state["plan_steps"],
            "findings_count": len(final_state["findings"]),
            "report": final_state["final_report"],
            "duration_seconds": dur,
            "metrics": {"total_duration_s": dur}
        }


def run_pipeline(query: str, verbose: bool = True) -> Dict[str, Any]:
    pipeline = UnifiedResearchPipeline()
    return pipeline.execute(query=query, verbose=verbose)
