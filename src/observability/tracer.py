"""Langfuse Observability & Tracing Module (Compatible with Langfuse v3 & v4)."""
import functools
import types
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

# Bridge compatibility for libraries checking langfuse.version and v4 init kwargs/methods
try:
    import langfuse
    if not hasattr(langfuse, "version"):
        langfuse.version = types.SimpleNamespace(__version__=getattr(langfuse, "__version__", "4.15.1"))
    if hasattr(langfuse, "Langfuse") and not getattr(langfuse.Langfuse, "_patched_kwargs", False):
        _orig_lf_init = langfuse.Langfuse.__init__
        def _adapted_lf_init(self, *args, **kwargs):
            kwargs.pop("sdk_integration", None)
            return _orig_lf_init(self, *args, **kwargs)
        _adapted_lf_init._patched_kwargs = True
        langfuse.Langfuse.__init__ = _adapted_lf_init

    class LangfuseSpanBridge:
        def __init__(self, client, **kwargs):
            self.client = client
            self.name = kwargs.get("name", "span")
            self.metadata = kwargs
            if hasattr(client, "create_event"):
                try:
                    client.create_event(name=f"Span:{self.name}", metadata=kwargs)
                except Exception:
                    pass

        def end(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):
            pass

        def score(self, *args, **kwargs):
            pass

    class LangfuseTraceBridge:
        def __init__(self, client, **kwargs):
            self.client = client
            self.id = kwargs.get("id", "trace-id")
            self.name = kwargs.get("name", "default")
            if hasattr(client, "create_event"):
                try:
                    client.create_event(name=f"Trace:{self.name}", metadata=kwargs)
                except Exception:
                    pass

        def generation(self, **kwargs):
            if hasattr(self.client, "create_event"):
                try:
                    self.client.create_event(name=f"Generation:{kwargs.get('name', 'gen')}", metadata=kwargs)
                except Exception:
                    pass
            return LangfuseSpanBridge(self.client, **kwargs)

        def span(self, **kwargs):
            return LangfuseSpanBridge(self.client, **kwargs)

        def update(self, *args, **kwargs):
            pass

        def score(self, *args, **kwargs):
            pass

        def event(self, *args, **kwargs):
            pass

    if hasattr(langfuse, "Langfuse") and not hasattr(langfuse.Langfuse, "trace"):
        langfuse.Langfuse.trace = lambda self, **kw: LangfuseTraceBridge(self, **kw)
except Exception:
    pass

from src.common.config import settings


class TraceRecord:
    """Trace recording object that mirrors traces locally and exports to Langfuse Cloud."""

    def __init__(self, name: str, trace_id: str, session_id: Optional[str] = None, client: Any = None):
        self.name = name
        self.trace_id = trace_id
        self.session_id = session_id or str(uuid.uuid4())
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.spans: List[Dict[str, Any]] = []
        self.generations: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.tags: List[str] = []
        self.client = client
        self.trace_url: Optional[str] = None

    def span(self, name: str, input_data: Any = None, metadata: Optional[Dict[str, Any]] = None):
        span_id = str(uuid.uuid4())
        start = time.time()
        span_dict = {
            "id": span_id,
            "name": name,
            "input": input_data,
            "start_time": start,
            "metadata": metadata or {},
        }
        self.spans.append(span_dict)

        class SpanContext:
            def __enter__(self_ctx):
                return span_dict

            def __exit__(self_ctx, exc_type, exc_val, exc_tb):
                span_dict["end_time"] = time.time()
                span_dict["duration_seconds"] = round(span_dict["end_time"] - start, 4)
                if exc_val:
                    span_dict["error"] = str(exc_val)
                    span_dict["status"] = "ERROR"
                else:
                    span_dict["status"] = "SUCCESS"

                # Send observation event to Langfuse Cloud if live
                if self.client:
                    try:
                        self.client.create_event(
                            name=f"Span:{name}",
                            metadata={
                                "span_id": span_id,
                                "duration_seconds": span_dict["duration_seconds"],
                                "status": span_dict["status"],
                                **span_dict["metadata"]
                            },
                            input=input_data,
                            output=span_dict.get("output", "")
                        )
                    except Exception:
                        pass

        return SpanContext()

    def log_generation(self, name: str, model: str, prompt: Any, completion: Any, usage: Dict[str, int]):
        self.generations.append({
            "name": name,
            "model": model,
            "prompt": prompt,
            "completion": completion,
            "usage": usage,
            "timestamp": time.time(),
        })

        if self.client:
            try:
                self.client.create_event(
                    name=f"Generation:{name}",
                    metadata={"model": model, "usage": usage},
                    input=prompt,
                    output=completion
                )
            except Exception:
                pass

    def end(self, output: Any = None, status: str = "SUCCESS"):
        self.end_time = time.time()
        self.metadata["output"] = output
        self.metadata["status"] = status
        self.metadata["total_duration_seconds"] = round(self.end_time - self.start_time, 4)

        if self.client:
            try:
                self.client.create_event(
                    name=f"TraceComplete:{self.name}",
                    metadata={
                        "trace_id": self.trace_id,
                        "session_id": self.session_id,
                        "status": status,
                        "tags": self.tags,
                        **self.metadata
                    },
                    output=output
                )
                self.client.flush()
                self.trace_url = f"https://cloud.langfuse.com/project/traces?search={self.trace_id}"
            except Exception as e:
                print(f"[Observability] Notice on trace flush: {e}")


class LangfuseTracer:
    """Manages Langfuse tracing, span lifecycle, generation metrics, and audit logs."""

    def __init__(self):
        self.client = None
        self.is_live = False
        self.active_traces: Dict[str, TraceRecord] = {}
        self._init_client()

    def _init_client(self):
        """Initialize Langfuse client with credentials."""
        if settings.LANGFUSE_ENABLED and settings.LANGFUSE_PUBLIC_KEY and not settings.LANGFUSE_PUBLIC_KEY.startswith("pk-lf-mock"):
            try:
                from langfuse import Langfuse
                self.client = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST or settings.LANGFUSE_BASE_URL
                )
                auth_ok = self.client.auth_check()
                if auth_ok:
                    self.is_live = True
                    print("[Observability] Langfuse Cloud Client connected & authenticated successfully.")
                else:
                    self.is_live = False
            except Exception as e:
                print(f"[Observability] Notice: Langfuse initialized in local audit mode: {e}")
                self.is_live = False
        else:
            self.is_live = False

    def create_trace(
        self,
        name: str,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TraceRecord:
        """Create a new trace context."""
        trace_id = str(uuid.uuid4())
        session = session_id or f"sess-{uuid.uuid4().hex[:8]}"

        local_trace = TraceRecord(
            name=name,
            trace_id=trace_id,
            session_id=session,
            client=self.client if self.is_live else None
        )
        local_trace.tags = tags or ["aipoc", "langgraph", "litellm"]
        local_trace.metadata = metadata or {}

        if self.is_live and self.client:
            try:
                self.client.create_event(
                    name=f"TraceStart:{name}",
                    metadata={
                        "trace_id": trace_id,
                        "session_id": session,
                        "tags": local_trace.tags,
                        **local_trace.metadata
                    }
                )
            except Exception as e:
                print(f"[Observability] Langfuse trace start warning: {e}")

        self.active_traces[trace_id] = local_trace
        self.active_traces[session] = local_trace
        return local_trace

    def flush(self):
        """Flush any pending events to Langfuse."""
        if self.is_live and self.client:
            try:
                self.client.flush()
            except Exception:
                pass


_tracer_instance: Optional[LangfuseTracer] = None


def get_tracer() -> LangfuseTracer:
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = LangfuseTracer()
    return _tracer_instance


def observe_step(step_name: str):
    """Decorator to trace individual function executions."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                raise e
        return wrapper
    return decorator
