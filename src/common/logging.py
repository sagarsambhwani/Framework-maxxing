"""Rich Colored Terminal Logging & Console Visualization.

Provides standardized, ANSI-colored, timestamped logging for all system events:
    - Incoming web / agent requests
    - NeMo Guardrail validation outcomes (Passed vs Blocked)
    - Multi-provider gateway routing decisions (Groq, Gemini, OpenRouter)
    - Streaming latency metrics (TTFT, Total Duration, Tokens/sec)
    - Langfuse cloud observability synchronization events
    - Granular function-level debug tracing (when DEBUG_MODE=True)

Why this module exists:
    Streamlit and background frameworks often buffer standard stdout or bury
    logs in UI widgets. This logger guarantees unbuffered, immediate, and
    visually structured output in the developer's terminal.
"""

import sys
import time

# Ensure UTF-8 output encoding across Windows PowerShell and CMD terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class Colors:
    """ANSI terminal escape color codes for structured log output."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[35m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def term_log(tag: str, message: str, color: str = Colors.CYAN):
    """Outputs a timestamped, color-coded log message immediately to terminal stdout.

    Args:
        tag: Category identifier (e.g. '[REQUEST]', '[GUARDRAIL]', '[ROUTER]', '[STREAM]').
        message: Informational details, latency, model names, or token speeds.
        color: ANSI color code from the Colors class (defaults to Cyan).
    """
    timestamp = time.strftime("%H:%M:%S")
    # flush=True ensures real-time emission without OS buffer delays
    print(f"{Colors.BOLD}[{timestamp}]{Colors.END} {color}{tag}{Colors.END} {message}", flush=True)


def debug_log(tag: str, message: str):
    """Outputs granular debug diagnostic info when DEBUG_MODE is active.

    Args:
        tag: Diagnostic category (e.g. '[DEBUG:STATE]', '[DEBUG:PAYLOAD]').
        message: Detailed JSON or internal variable state dump.
    """
    from src.common.config import settings
    if settings.DEBUG_MODE:
        timestamp = time.strftime("%H:%M:%S")
        print(f"{Colors.GRAY}[{timestamp}] {Colors.MAGENTA}{tag}{Colors.GRAY} {message}{Colors.END}", flush=True)


def print_banner(title: str, subtitle: str = ""):
    """Renders a visual CLI banner demarcating major run modes or pipeline starts.

    Args:
        title: Main header text.
        subtitle: Optional secondary description or session details.
    """
    print("\n" + "=" * 80, flush=True)
    print(f"{Colors.BOLD}{Colors.GREEN}🚀 {title}{Colors.END}", flush=True)
    if subtitle:
        print(f"   {Colors.CYAN}{subtitle}{Colors.END}", flush=True)
    print("=" * 80 + "\n", flush=True)
