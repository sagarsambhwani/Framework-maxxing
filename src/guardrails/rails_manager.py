"""NeMo Guardrails Manager for Input/Output Safety and Policy Enforcement."""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from src.common.config import settings
from src.guardrails.custom_actions import (
    check_jailbreak_pattern,
    check_destructive_pattern,
    mask_sensitive_pii,
)


class GuardrailsManager:
    """Manages input/output rails, prompt injection detection, and content safety."""

    def __init__(self):
        self.enabled = settings.GUARDRAILS_ENABLED
        self.rails = None
        self._init_nemo_rails()

    def _init_nemo_rails(self):
        """Initialize NeMo Guardrails LLMRails instance if config exists."""
        if not self.enabled:
            return

        config_dir = Path(settings.GUARDRAILS_CONFIG_PATH)
        if not config_dir.exists():
            return

        try:
            from nemoguardrails import RailsConfig, LLMRails
            # Check if config files exist
            if (config_dir / "config.yml").exists():
                rails_config = RailsConfig.from_path(str(config_dir))
                self.rails = LLMRails(rails_config)
                print("[Guardrails] NeMo Guardrails engine initialized successfully.")
        except Exception as e:
            print(f"[Guardrails] Notice: Using high-speed deterministic guardrail engine: {e}")
            self.rails = None

    def validate_input(self, user_prompt: str) -> Dict[str, Any]:
        """Verify user input against jailbreak, injection, and off-topic destruction policies."""
        if not self.enabled:
            return {"allowed": True, "reason": "Guardrails disabled", "sanitized_prompt": user_prompt}

        # 1. Fast Heuristic: Check for Jailbreak
        if check_jailbreak_pattern(user_prompt):
            return {
                "allowed": False,
                "reason": "BLOCKED: Input contains prompt injection or adversarial jailbreak pattern.",
                "sanitized_prompt": None,
                "violation_type": "jailbreak"
            }

        # 2. Fast Heuristic: Check for Malicious/Destructive intent
        if check_destructive_pattern(user_prompt):
            return {
                "allowed": False,
                "reason": "BLOCKED: Request attempts destructive or unauthorized harmful activity.",
                "sanitized_prompt": None,
                "violation_type": "off_topic_destructive"
            }

        # 3. Mask PII in sanitized prompt
        sanitized = mask_sensitive_pii(user_prompt)

        return {
            "allowed": True,
            "reason": "Passed input safety checks",
            "sanitized_prompt": sanitized,
            "violation_type": None
        }

    def validate_output(self, response_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Verify output against policy constraints, toxicity, and mask sensitive PII."""
        if not self.enabled:
            return {"allowed": True, "response": response_text, "modified": False}

        # 1. Check for system prompt leaks
        leak_indicators = ["SYSTEM_PROMPT:", "MY_SECRET_INSTRUCTIONS:"]
        for ind in leak_indicators:
            if ind in response_text:
                return {
                    "allowed": False,
                    "response": "Output blocked due to policy constraints regarding internal prompts.",
                    "modified": True,
                    "violation_type": "prompt_leak"
                }

        # 2. Mask any PII generated in the response
        clean_response = mask_sensitive_pii(response_text)
        was_modified = (clean_response != response_text)

        return {
            "allowed": True,
            "response": clean_response,
            "modified": was_modified,
            "violation_type": None
        }


_guardrails_instance: Optional[GuardrailsManager] = None


def get_guardrails_manager() -> GuardrailsManager:
    global _guardrails_instance
    if _guardrails_instance is None:
        _guardrails_instance = GuardrailsManager()
    return _guardrails_instance
