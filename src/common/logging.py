"""Rich Colored Terminal Logging Utilities."""
import sys
import time

# Ensure UTF-8 on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[35m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def term_log(tag: str, message: str, color: str = Colors.CYAN):
    """Prints a styled, timestamped log line immediately to terminal."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"{Colors.BOLD}[{timestamp}]{Colors.END} {color}{tag}{Colors.END} {message}", flush=True)


def print_banner(title: str, subtitle: str = ""):
    """Prints a clean CLI banner."""
    print("\n" + "=" * 80, flush=True)
    print(f"{Colors.BOLD}{Colors.GREEN}🚀 {title}{Colors.END}", flush=True)
    if subtitle:
        print(f"   {Colors.CYAN}{subtitle}{Colors.END}", flush=True)
    print("=" * 80 + "\n", flush=True)
