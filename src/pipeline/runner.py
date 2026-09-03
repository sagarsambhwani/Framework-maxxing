"""Unified End-to-End Pipeline Orchestrator.

This module provides the `UnifiedResearchPipeline` entrypoint that binds:
    1. NeMo Guardrails Input Validation
    2. LangGraph Autonomous Research Planner & Tool Execution
    3. Multi-Provider Gateway Routing (Groq, Gemini, OpenRouter)
    4. Langfuse Cloud Observability Synchronization

Usage:
    from src.pipeline.runner import run_pipeline
    result = run_pipeline("Evaluate multi-cloud LLM gateway latency")
"""

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
    """Orchestrates end-to-end execution of secure, multi-model research requests."""

    def execute(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """Runs the research query through the protected autonomous pipeline.

        Args:
            query: User research question text.
            verbose: If True, prints formatted console status logs.

        Returns:
            Dict containing:
                - 'session_id': Session correlation ID.
                - 'status': 'SUCCESS' or 'BLOCKED'.
                - 'plan_steps': List of planned research tasks.
                - 'findings_count': Number of executed tool steps.
                - 'report': Final markdown executive report.
                - 'duration_seconds': Total execution duration.
                - 'metrics': Telemetry dictionary.
        """
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

        # Invoke the compiled LangGraph workflow
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
    """Helper function to instantiate and execute the unified research pipeline.

    Args:
        query: User input query.
        verbose: Verbose terminal logging flag.

    Returns:
        Execution result dictionary.
    """
    pipeline = UnifiedResearchPipeline()
    return pipeline.execute(query=query, verbose=verbose)
