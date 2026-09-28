"""Zero-Trust Compliance & PII/PCI-DSS Sanitizer Gate for Banking Records.

Guarantees that sensitive payment data, credit cards, bank account numbers,
and customer PII are redacted before embedding generation and vector storage.
"""

import re
from typing import Tuple, List

class ZeroTrustComplianceSanitizer:
    """Pre-inference & pre-index privacy redaction gate."""

    # Regex patterns for high-risk banking PII
    CREDIT_CARD_REGEX = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
    SSN_REGEX = re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b')
    EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
    ACCOUNT_REGEX = re.compile(r'\b(?:Account|ACCT|IBAN)[\s#:]+([A-Za-z0-9]{8,24})\b', re.IGNORECASE)

    # Pathological & Evasion Artifacts: Null bytes, Zero-Width Spaces, BOM, and BiDi Overrides
    NULL_BYTE_REGEX = re.compile(r'\x00')
    ZERO_WIDTH_REGEX = re.compile(r'[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E\u2066-\u2069]')

    @classmethod
    def sanitize(cls, text: str) -> Tuple[str, List[str]]:
        """Redacts sensitive financial PII and purges pathological/evasion characters.

        Args:
            text: Raw input text.

        Returns:
            Tuple of (sanitized_text, list of redacted field names).
        """
        redacted_fields: List[str] = []
        sanitized = text

        # 0. Pathological Input Defense: Strip Null Bytes & Zero-Width Evasion Chars
        if cls.NULL_BYTE_REGEX.search(sanitized):
            redacted_fields.append("NULL_BYTE_INJECTION")
            sanitized = cls.NULL_BYTE_REGEX.sub('', sanitized)

        if cls.ZERO_WIDTH_REGEX.search(sanitized):
            redacted_fields.append("ZERO_WIDTH_EVASION")
            sanitized = cls.ZERO_WIDTH_REGEX.sub('', sanitized)

        # 1. PCI-DSS Credit Card Masking (Keep last 4 digits for audit context)
        def _mask_cc(match):
            raw = re.sub(r'[\s-]', '', match.group(0))
            if 13 <= len(raw) <= 19:
                redacted_fields.append("PCI_DSS_CREDIT_CARD")
                return f"[MASKED_PAN_{raw[-4:]}]"
            return match.group(0)

        sanitized = cls.CREDIT_CARD_REGEX.sub(_mask_cc, sanitized)

        # 2. SSN / National Identity Numbers
        if cls.SSN_REGEX.search(sanitized):
            redacted_fields.append("SSN_NATIONAL_ID")
            sanitized = cls.SSN_REGEX.sub("[MASKED_SSN]", sanitized)

        # 3. Bank Account Numbers
        def _mask_account(match):
            redacted_fields.append("BANK_ACCOUNT_NUMBER")
            raw_acc = match.group(1)
            return match.group(0).replace(raw_acc, f"[MASKED_ACC_{raw_acc[-4:]}]")

        sanitized = cls.ACCOUNT_REGEX.sub(_mask_account, sanitized)

        # 4. Emails
        if cls.EMAIL_REGEX.search(sanitized):
            redacted_fields.append("CUSTOMER_EMAIL")
            sanitized = cls.EMAIL_REGEX.sub("[MASKED_EMAIL]", sanitized)

        # 5. Phone Numbers
        if cls.PHONE_REGEX.search(sanitized):
            redacted_fields.append("CUSTOMER_PHONE")
            sanitized = cls.PHONE_REGEX.sub("[MASKED_PHONE]", sanitized)

        return sanitized, list(set(redacted_fields))

# Singleton instance
sanitizer = ZeroTrustComplianceSanitizer()
