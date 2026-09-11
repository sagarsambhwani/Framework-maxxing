"""Contextual Chunking Engine (Anthropic-style Context Injection).

Prepends document-level metadata and section breadcrumbs to each chunk
so that embedded vectors retain the overarching document identity and domain context.
"""

from typing import Dict, Any

class ContextualChunker:
    """Injects contextual summaries into chunk payloads prior to vectorization."""

    @classmethod
    def inject_context(
        cls,
        chunk_text: str,
        doc_title: str,
        doc_type: str,
        section_header: str
    ) -> str:
        """Injects a standardized contextual header into the chunk.

        Args:
            chunk_text: The raw chunk text.
            doc_title: Document title or filename.
            doc_type: Document classification category.
            section_header: The enclosing section title.

        Returns:
            Context-enriched string.
        """
        context_header = f"[Context: Document='{doc_title}' | Domain='{doc_type}' | Section='{section_header}']"
        return f"{context_header}\n{chunk_text.strip()}"

    @classmethod
    def build_breadcrumb(
        cls,
        doc_title: str,
        domain: str,
        section_path: list
    ) -> str:
        path_str = " > ".join(section_path)
        return f"[{domain}] {doc_title} > {path_str}"

# Singleton instance
contextual_chunker = ContextualChunker()
