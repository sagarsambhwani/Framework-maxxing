"""Langfuse Cloud Observability Tracer & Lifecycle Manager."""
import types
from typing import Any, Dict, Optional

from src.common.config import settings
from src.common.logging import term_log, Colors

# Langfuse v4 compatibility bridge
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
        self.client = None
        if settings.LANGFUSE_ENABLED and settings.LANGFUSE_PUBLIC_KEY and not settings.LANGFUSE_PUBLIC_KEY.startswith("pk-lf-mock"):
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
                term_log("📈 [LANGFUSE]", f"Local fallback mode: {e}", Colors.YELLOW)

    def log_event(self, name: str, session_id: str, metadata: Dict[str, Any], input_data: Any = None, output_data: Any = None):
        """Creates a trace event and flushes to Langfuse Cloud."""
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


# Global tracer singleton
tracer = ObservabilityTracer()
