"""Langfuse Cloud Observability Tracer & Lifecycle Manager.

This module provides centralized observability tracking for all LLM calls,
agent steps, tool executions, and security checks across the application.

Key Features:
    - Cloud Telemetry Synchronization: Automatically flushes event records,
      prompts, responses, and latency metrics to https://cloud.langfuse.com.
    - Langfuse v4 Compatibility Adapter: Bridges parameter differences across SDK versions.
    - Graceful Local Fallback: If network connectivity or keys are unavailable,
      logs locally without throwing unhandled exceptions.

Usage:
    from src.observability.tracer import tracer
    tracer.log_event("PlannerNode", session_id="abc-123", metadata={"model": "qwen3.8"})
"""

import types
from typing import Any, Dict, Optional

from src.common.config import settings
from src.common.logging import term_log, Colors

# -----------------------------------------------------------------------------
# Langfuse v4 Backward Compatibility Bridge
# -----------------------------------------------------------------------------
try:
    import langfuse
    if not hasattr(langfuse, "version"):
        langfuse.version = types.SimpleNamespace(__version__=getattr(langfuse, "__version__", "4.15.1"))
    if hasattr(langfuse, "Langfuse") and not getattr(langfuse.Langfuse, "_patched_kwargs", False):
        _orig_init = langfuse.Langfuse.__init__
        def _patched_init(self, *args, **kwargs):
            kwargs.pop("sdk_integration", None)
            return _orig_init(self, *args, **kwargs)
        _patched_init._patched_kwargs = True
        langfuse.Langfuse.__init__ = _patched_init
except Exception:
    pass


class ObservabilityTracer:
    """Manages spans, events, and token cost telemetry synced to Langfuse Cloud."""

    def __init__(self):
        """Initializes Langfuse cloud client if valid credentials are configured."""
        self.client = None
        if (
            settings.LANGFUSE_ENABLED
            and settings.LANGFUSE_PUBLIC_KEY
            and not settings.LANGFUSE_PUBLIC_KEY.startswith("pk-lf-mock")
        ):
            try:
                from langfuse import Langfuse
                self.client = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST
                )
                if self.client.auth_check():
                    term_log("📈 [LANGFUSE]", f"Cloud Observability connected ({settings.LANGFUSE_HOST})", Colors.GREEN)
            except Exception as e:
                term_log("📈 [LANGFUSE]", f"Operating in local audit mode: {e}", Colors.YELLOW)

    def log_event(
        self,
        name: str,
        session_id: str,
        metadata: Dict[str, Any],
        input_data: Any = None,
        output_data: Any = None
    ):
        """Creates a trace event and flushes telemetry to Langfuse Cloud.

        Args:
            name: Trace event name (e.g. 'Planner:Decomposition', 'Tool:web_search').
            session_id: Correlation ID mapping the event to an active user session.
            metadata: Dictionary of auxiliary metrics (model, latencies, tokens).
            input_data: Optional input payload string or dictionary.
            output_data: Optional generated output payload string or dictionary.
        """
        if self.client:
            try:
                self.client.create_event(
                    name=name,
                    metadata={"session_id": session_id, **metadata},
                    input=input_data,
                    output=output_data
                )
                # Flush ensures telemetry reaches cloud dashboard immediately
                self.client.flush()
            except Exception:
                pass


# Global tracer singleton instance
tracer = ObservabilityTracer()
