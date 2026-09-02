"""Custom Actions for NeMo Guardrails."""
import re
from typing import Dict, Any, Optional


def check_jailbreak_pattern(text: str) -> bool:
    """Deterministic regex-based heuristic for jailbreak/injection detection."""
    jailbreak_patterns = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"developer\s+mode\s+(enabled|on)",
        r"dan\s+mode",
        r"system\s+prompt\s+(verbatim|reveal|leak|print)",
        r"bypass\s+(safety|content\s+policy|guardrails)",
        r"do\s+anything\s+now",
        r"you\s+are\s+an\s+unrestricted\s+ai"
    ]
    for pattern in jailbreak_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def check_destructive_pattern(text: str) -> bool:
    """Detect destructive, malicious cyberattack, or weaponization requests."""
    destructive_patterns = [
        r"(build|create|manufacture)\s+(a\s+)?(bomb|explosive|weapon|virus|malware)",
        r"(hack|breach|infiltrate)\s+(into\s+)?(bank|system|database|server)",
        r"(format|erase|delete)\s+(c:|root|hard\s*drive|all\s+files)",
        r"ransomware\s+payload"
    ]
    for pattern in destructive_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def mask_sensitive_pii(text: str) -> str:
    """Mask email addresses, phone numbers, and SSNs."""
    # Mask emails
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", text)
    # Mask phone numbers
    text = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]", text)
    # Mask SSN
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", text)
    return text
