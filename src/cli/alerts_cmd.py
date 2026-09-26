"""CLI command handler for system observability, Phoenix visual tracing, and SLA alerting."""

import os
import sys
import json
import subprocess
from src.common.logging import term_log, print_banner, Colors


def run_phoenix_server():
    """Launches local Arize Phoenix visual tracing UI on http://localhost:6006."""
    print_banner("ARIZE PHOENIX LOCAL OBSERVABILITY SERVER", "Dashboard: http://localhost:6006 | OpenTelemetry: ACTIVE")
    print("✓ Starting Arize Phoenix at http://localhost:6006...")
    print("Press Ctrl+C to stop the Phoenix server.\n")
    env = os.environ.copy()
    env["PHOENIX_PORT"] = "6006"
    env["PHOENIX_HOST"] = "127.0.0.1"
    try:
        subprocess.run([sys.executable, "-m", "phoenix.server.main", "serve"], env=env)
    except KeyboardInterrupt:
        print("\nStopping Arize Phoenix server...")
    except Exception as e:
        term_log("❌ [PHOENIX ERROR]", f"Could not launch Phoenix: {e}", Colors.RED)


def run_alert_check():
    """Scans the latest local trace for production SLA violations."""
    from src.observability.alerts import ProductionAlertEngine
    latest_file = os.path.join(os.getcwd(), "traces", "latest_trace.json")
    if not os.path.exists(latest_file):
        term_log("⚠️ [ALERTS]", "No traces found in 'traces/latest_trace.json'. Run a workflow first.", Colors.YELLOW)
        return

    with open(latest_file, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    print_banner("PRODUCTION TRACE SLA ALERT SCANNER", f"Scanning Trace ID: {trace_data.get('trace_id')} | Session: {trace_data.get('session_id')}")
    alerts = ProductionAlertEngine.evaluate_trace(trace_data)
    ProductionAlertEngine.render_alerts(alerts)
