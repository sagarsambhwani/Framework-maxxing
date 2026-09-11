"""Document Triage & Classification Microservice.

Classifies incoming banking & enterprise documents into specialized domain types:
    - LEGAL_REGULATION: Banking laws, compliance statutes, Dodd-Frank, Basel III, 12 CFR.
    - INVOICE_RECEIPT: Vendor bills, tax invoices, purchase orders, fee slips.
    - BANK_STATEMENT: Monthly account statements, ledgers, transaction histories.
    - GENERAL_DOC: Technical memos, architectural RFCs, policy guidelines.
"""

import re
from typing import Dict, Any, List, Tuple

class DocumentTriage:
    """Document classification and layout extraction engine."""

    LEGAL_KEYWORDS = [
        "section", "pursuant to", "regulation", "statute", "subparagraph",
        "dodd-frank", "basel", "12 cfr", "article", "clause", "federal register",
        "compliance", "act of", "amendment", "jurisdiction"
    ]

    INVOICE_KEYWORDS = [
        "invoice", "bill to", "tax invoice", "subtotal", "amount due",
        "invoice date", "invoice no", "vat", "gst", "remit to", "line items",
        "purchase order", "po number", "due date"
    ]

    STATEMENT_KEYWORDS = [
        "account statement", "balance brought forward", "ending balance",
        "transaction history", "debit", "credit", "ledger balance", "cleared balance",
        "account number", "statement period", "deposits", "withdrawals"
    ]

    @classmethod
    def classify(cls, text: str, filename: str = "") -> Tuple[str, float]:
        """Classifies document type based on keyword density and layout signals.

        Args:
            text: Raw document text.
            filename: Original filename.

        Returns:
            Tuple of (doc_type, confidence_score).
        """
        lower = (text[:3000] + " " + filename).lower()

        legal_score = sum(1 for kw in cls.LEGAL_KEYWORDS if kw in lower)
        invoice_score = sum(1 for kw in cls.INVOICE_KEYWORDS if kw in lower)
        statement_score = sum(1 for kw in cls.STATEMENT_KEYWORDS if kw in lower)

        scores = {
            "LEGAL_REGULATION": legal_score,
            "INVOICE_RECEIPT": invoice_score,
            "BANK_STATEMENT": statement_score
        }

        best_type = max(scores, key=scores.get)
        best_count = scores[best_type]

        if best_count >= 2:
            confidence = min(0.99, round(0.60 + (best_count * 0.08), 2))
            return best_type, confidence

        return "GENERAL_DOC", 0.70

    @classmethod
    def extract_layout(cls, raw_text: str) -> Dict[str, Any]:
        """Extracts structural hierarchy: headers, tables, and paragraphs.

        Args:
            raw_text: Raw plain text or markdown.

        Returns:
            Dict containing title, sections list, and detected tables.
        """
        lines = raw_text.splitlines()
        sections = []
        tables = []
        current_section = {"header": "Preamble", "content": []}
        in_table = False
        table_buffer = []

        for line in lines:
            stripped = line.strip()

            # Detect Markdown Table
            if "|" in stripped and stripped.startswith("|") and stripped.endswith("|"):
                in_table = True
                table_buffer.append(stripped)
                continue
            elif in_table:
                in_table = False
                if len(table_buffer) >= 2:
                    table_md = "\n".join(table_buffer)
                    tables.append(table_md)
                    current_section["content"].append(table_md)
                table_buffer = []

            # Detect Section Header (Markdown # or numbered legal sections e.g. "Section 4.1")
            header_match = re.match(r'^(?:#+\s*(.+)|(?:Section|Article)\s*([0-9A-Za-z\.\-\(\)]+[:\.\s].*))$', stripped, re.IGNORECASE)
            if header_match:
                if current_section["content"]:
                    sections.append({
                        "header": current_section["header"],
                        "text": "\n".join(current_section["content"])
                    })
                header_title = header_match.group(1) or header_match.group(2) or stripped
                current_section = {"header": header_title.strip(), "content": []}
            else:
                if stripped:
                    current_section["content"].append(stripped)

        # Append last section
        if current_section["content"]:
            sections.append({
                "header": current_section["header"],
                "text": "\n".join(current_section["content"])
            })

        return {
            "total_sections": len(sections),
            "sections": sections,
            "tables_found": len(tables),
            "tables": tables
        }

# Singleton instance
triage_engine = DocumentTriage()
