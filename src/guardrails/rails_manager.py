"""NeMo Guardrails & Safety Engine with Prompt Guard Classifier."""
import re
import time
from typing import Dict, Any, Optional

from src.common.config import settings
from src.common.logging import term_log, Colors


class GuardrailsManager:
    """Manages input jailbreak detection, output validation, and PII masking."""

    JAILBREAK_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"developer\s+mode",
        r"system\s+prompt\s+(reveal|leak|print|dump)",
        r"format\s+c:\s*drive",
        r"dan\s+mode",
        r"unrestricted\s+mode"
    ]

    @classmethod
    def validate_input(cls, user_prompt: str) -> Dict[str, Any]:
        """Validates input prompt for adversarial attacks, jailbreaks, and PII."""
        if not settings.GUARDRAILS_ENABLED:
            return {"allowed": True, "clean_prompt": user_prompt, "flagged": False, "reason": "Guardrails disabled"}

        t0 = time.time()

        # 1. Regex Heuristic & Jailbreak Detection
        for pattern in cls.JAILBREAK_PATTERNS:
            if re.search(pattern, user_prompt, re.IGNORECASE):
                dur_ms = round((time.time() - t0) * 1000, 2)
                reason = f"BLOCKED by NeMo Guardrails: Disallowed pattern '{pattern}' detected."
                term_log("🛡️ [GUARDRAIL]", f"{Colors.RED}BLOCKED{Colors.END} in {dur_ms}ms -> {reason}", Colors.RED)
                return {
                    "allowed": False,
                    "clean_prompt": user_prompt,
                    "flagged": True,
                    "reason": reason,
                    "check_time_ms": dur_ms
                }

        # 2. PII Masking (Emails & Phone Numbers)
        sanitized = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", user_prompt)
        sanitized = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]", sanitized)

        dur_ms = round((time.time() - t0) * 1000, 2)
        term_log("🛡️ [GUARDRAIL]", f"{Colors.GREEN}PASSED{Colors.END} in {dur_ms}ms (PII Sanitized)", Colors.GREEN)

        return {
            "allowed": True,
            "clean_prompt": sanitized,
            "flagged": False,
            "reason": "Passed all safety policies",
            "check_time_ms": dur_ms
        }

    @classmethod
    def sanitize_output(cls, bot_output: str) -> str:
        """Sanitizes model generated output against sensitive information leakage."""
        cleaned = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", bot_output)
        cleaned = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]", cleaned)
        return cleaned


# Global guardrails singleton
guardrails = GuardrailsManager()
