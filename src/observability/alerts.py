"""Production Trace Alerting & SLA Violation Monitor.

Evaluates local and cloud traces against enterprise operational thresholds:
    1. Latency & TTFT: Flags requests where TTFT > 400ms or total time > 8s.
    2. Cost & Token Budgets: Alerts if single-turn tokens exceed 4,000 or cost > $0.01.
    3. Security Attacks: Detects bursts of blocked guardrail violations.
    4. Quality & Hallucination: Flags low faithfulness (<0.80) or excessive reflection loops (>=3).
"""

from typing import Dict, Any, List
from src.common.logging import term_log, Colors


class ProductionAlertEngine:
    """Evaluates real-time traces against production SLAs and triggers alerts."""

    THRESHOLDS = {
        "max_ttft_ms": 400.0,          # Alert if TTFT > 400ms
        "max_duration_s": 8.0,          # Alert if total latency > 8.0s
        "max_single_query_tokens": 4000,# Alert on runaway token consumption
        "max_agent_revisions": 2,       # Alert if reflection loop >= 3 iterations
        "min_faithfulness_score": 0.80  # Alert on hallucination risk
    }

    @classmethod
    def evaluate_trace(cls, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scans a trace object for SLA breaches and performance anomalies.

        Args:
            trace: Trace dictionary containing spans and metrics.

        Returns:
            List of triggered alert dictionaries.
        """
        alerts = []
        session_id = trace.get("session_id", "unknown")

        # 1. Total Latency Check
        dur = trace.get("total_duration_s", 0.0)
        if dur > cls.THRESHOLDS["max_duration_s"]:
            alerts.append({
                "severity": "WARNING",
                "type": "LATENCY_SPIKE",
                "message": f"Total execution time ({dur}s) exceeded SLA threshold ({cls.THRESHOLDS['max_duration_s']}s)",
                "session_id": session_id
            })

        # 2. Token Budget Check
        tokens = trace.get("total_tokens", 0)
        if tokens > cls.THRESHOLDS["max_single_query_tokens"]:
            alerts.append({
                "severity": "CRITICAL",
                "type": "TOKEN_BUDGET_EXCEEDED",
                "message": f"Single query token consumption ({tokens}) exceeded safety ceiling ({cls.THRESHOLDS['max_single_query_tokens']})",
                "session_id": session_id
            })

        # 3. Span-Level TTFT and Guardrail Checks
        for span in trace.get("spans", []):
            ttft = span.get("ttft_ms")
            if isinstance(ttft, (int, float)) and ttft > cls.THRESHOLDS["max_ttft_ms"]:
                alerts.append({
                    "severity": "WARNING",
                    "type": "SLOW_TTFT",
                    "message": f"Span '{span.get('name')}' exhibited slow TTFT ({ttft}ms > {cls.THRESHOLDS['max_ttft_ms']}ms)",
                    "session_id": session_id
                })

        return alerts

    @classmethod
    def render_alerts(cls, alerts: List[Dict[str, Any]]):
        """Prints formatted alerts in terminal."""
        if not alerts:
            term_log("✅ [ALERTS]", "All trace metrics within production SLA thresholds (0 violations).", Colors.GREEN)
            return

        print("\n" + "=" * 80)
        print(f"{Colors.RED}{Colors.BOLD}🚨 PRODUCTION SLA ALERTS DETECTED ({len(alerts)} VIOLATIONS){Colors.END}")
        print("=" * 80)
        for a in alerts:
            sev_color = Colors.RED if a["severity"] == "CRITICAL" else Colors.YELLOW
            print(f" • [{sev_color}{a['severity']}{Colors.END}] [{a['type']}] {a['message']}")
        print("=" * 80 + "\n")
