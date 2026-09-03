"""Dual-Mode Local & Cloud Observability Tracer.

This module provides 100% transparent, self-contained, local JSON tracing
along with optional Langfuse Cloud synchronization.

Key Features:
    - Zero-Dependency Local JSON Traces: Automatically persists detailed, structured
      traces to `traces/<session_id>.json` and `traces/latest_trace.json`.
    - Granular Telemetry: Tracks Time-to-First-Token (TTFT), Prompt Tokens,
      Completion Tokens, Total Tokens, Latencies (ms), Cost ($), and I/O Payloads.
    - Cloud Telemetry Synchronization: Flushes traces to https://cloud.langfuse.com
      when configured.

Usage:
    from src.observability.tracer import tracer
    tracer.start_trace(session_id="abc-123", name="MarketingWorkflow")
    tracer.log_span(session_id="abc-123", name="Strategist", model="qwen3.8", ttft_ms=140, prompt_tokens=180, completion_tokens=600, duration_s=1.2)
    tracer.finalize_trace(session_id="abc-123")
"""

import os
import json
import time
import types
from typing import Any, Dict, Optional, List

from src.common.config import settings
from src.common.logging import term_log, debug_log, Colors

# Ensure traces directory exists
TRACES_DIR = os.path.join(os.getcwd(), "traces")
os.makedirs(TRACES_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Langfuse v4 Backward Compatibility Bridge (Optional)
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
    """Manages local JSON trace files and optional Langfuse Cloud telemetry."""

    def __init__(self):
        """Initializes in-memory session buffer and optional Langfuse client."""
        self.active_traces: Dict[str, Dict[str, Any]] = {}
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
            except Exception:
                pass

    def start_trace(self, session_id: str, name: str = "ExecutionTrace", metadata: Optional[Dict[str, Any]] = None):
        """Initializes a new local trace session container."""
        self.active_traces[session_id] = {
            "trace_id": f"trace-{session_id}",
            "name": name,
            "session_id": session_id,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "_start_t": time.time(),
            "total_duration_s": 0.0,
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "estimated_cost_usd": 0.0,
            "metadata": metadata or {},
            "spans": []
        }

    def log_event(
        self,
        name: str,
        session_id: str,
        metadata: Dict[str, Any],
        input_data: Any = None,
        output_data: Any = None
    ):
        """Logs a trace span to the local trace buffer and optionally syncs to Langfuse."""
        if session_id not in self.active_traces:
            self.start_trace(session_id, name=name)

        trace = self.active_traces[session_id]
        
        # Extract token and latency metrics
        prompt_tokens = metadata.get("prompt_tokens", 0)
        completion_tokens = metadata.get("tokens", metadata.get("completion_tokens", 0))
        total_tokens = prompt_tokens + completion_tokens
        duration_s = metadata.get("latency_s", metadata.get("duration_s", 0.0))
        ttft_ms = metadata.get("ttft_ms", metadata.get("ttft", "N/A"))
        model = metadata.get("model", "N/A")

        # Estimate standard Groq/OpenRouter pricing (~$0.05 / 1M tokens)
        cost_usd = round((total_tokens / 1_000_000) * 0.05, 6)

        span = {
            "span_id": f"span-{len(trace['spans']) + 1}",
            "name": name,
            "model": model,
            "duration_s": duration_s,
            "ttft_ms": ttft_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "metadata": metadata,
            "input": input_data,
            "output": output_data
        }

        trace["spans"].append(span)
        trace["total_tokens"] += total_tokens
        trace["total_prompt_tokens"] += prompt_tokens
        trace["total_completion_tokens"] += completion_tokens
        trace["estimated_cost_usd"] += cost_usd

        # Auto-persist local JSON file
        self._write_local_trace_file(session_id)

        # Sync to cloud if available
        if self.client:
            try:
                self.client.create_event(
                    name=name,
                    metadata={"session_id": session_id, **metadata},
                    input=input_data,
                    output=output_data
                )
                self.client.flush()
            except Exception:
                pass

    def finalize_trace(self, session_id: str) -> Dict[str, Any]:
        """Calculates total duration, saves final local JSON, and returns trace dictionary."""
        if session_id in self.active_traces:
            trace = self.active_traces[session_id]
            trace["total_duration_s"] = round(time.time() - trace.pop("_start_t", time.time()), 3)
            trace["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._write_local_trace_file(session_id)
            return trace
        return {}

    def _write_local_trace_file(self, session_id: str):
        """Writes formatted JSON trace files to disk (`traces/<session_id>.json` & `traces/latest_trace.json`)."""
        if session_id in self.active_traces:
            trace_data = dict(self.active_traces[session_id])
            # Remove private timing key if present
            trace_data.pop("_start_t", None)
            
            # 1. Write specific session trace
            session_file = os.path.join(TRACES_DIR, f"{session_id}.json")
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, indent=2)

            # 2. Write / overwrite latest trace pointer
            latest_file = os.path.join(TRACES_DIR, "latest_trace.json")
            with open(latest_file, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, indent=2)


# Global tracer singleton instance
tracer = ObservabilityTracer()
