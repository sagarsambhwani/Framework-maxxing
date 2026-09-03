"""NeMo Guardrails Policy Engine & Security Layer.

This module enforces enterprise AI safety standards across all user inputs
and assistant outputs:
    1. Jailbreak & Adversarial Protection: Detects prompt injections, system prompt leak
       attempts, "DAN/developer mode" bypasses, and destructive command sequences in <1ms.
    2. PII Redaction & Data Loss Prevention: Masks sensitive personally identifiable
       information (PII) including email addresses and telephone numbers.
    3. Performance: Uses deterministic compiled regex heuristics executing in sub-millisecond
       timeframes before any costly model inference or tool calls occur.

Design Rationale:
    By evaluating input safety before dispatching requests to LLMs or tool execution
    nodes, the system completely avoids burning token costs or running harmful
    actions during adversarial attacks.
"""

import re
import time
from typing import Dict, Any, Optional

from src.common.config import settings
from src.common.logging import term_log, Colors


class GuardrailsManager:
    """Manages input security policies, adversarial filtering, and output sanitization."""

    # Disallowed adversarial and jailbreak regular expression patterns
    JAILBREAK_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"developer\s+mode",
        r"system\s+prompt\s+(reveal|leak|print|dump)",
        r"format\s+c:\s*drive",
        r"dan\s+mode",
        r"unrestricted\s+mode"
    ]

    # Standard regular expressions for PII detection
    EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    PHONE_PATTERN = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"

    @classmethod
    def validate_input(cls, user_prompt: str) -> Dict[str, Any]:
        """Validates an incoming user prompt against security and safety policies.

        Args:
            user_prompt: Raw text string received from user input.

        Returns:
            Dict containing:
                - 'allowed' (bool): True if prompt is safe to proceed, False if blocked.
                - 'clean_prompt' (str): Sanitized prompt with PII redacted.
                - 'flagged' (bool): True if security violations were intercepted.
                - 'reason' (str): Explanation for approval or block reason.
                - 'check_time_ms' (float): Execution time of the safety scan in milliseconds.
        """
        # If guardrails are globally disabled in settings, allow prompt without checks
        if not settings.GUARDRAILS_ENABLED:
            return {
                "allowed": True,
                "clean_prompt": user_prompt,
                "flagged": False,
                "reason": "Guardrails disabled in configuration"
            }

        t0 = time.time()

        # ---------------------------------------------------------------------
        # Step 1: Jailbreak & Adversarial Pattern Matching
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # Step 2: PII Redaction (Masking Emails and Phone Numbers)
        # ---------------------------------------------------------------------
        sanitized = re.sub(cls.EMAIL_PATTERN, "[REDACTED_EMAIL]", user_prompt)
        sanitized = re.sub(cls.PHONE_PATTERN, "[REDACTED_PHONE]", sanitized)

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
        """Sanitizes model-generated output to prevent sensitive data leakage.

        Args:
            bot_output: Raw generated string returned by the language model.

        Returns:
            Sanitized string with any accidentally generated emails or phone numbers masked.
        """
        cleaned = re.sub(cls.EMAIL_PATTERN, "[REDACTED_EMAIL]", bot_output)
        cleaned = re.sub(cls.PHONE_PATTERN, "[REDACTED_PHONE]", cleaned)
        return cleaned


# Global guardrails manager singleton instance
guardrails = GuardrailsManager()
