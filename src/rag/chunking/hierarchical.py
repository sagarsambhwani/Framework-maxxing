"""Hierarchical Parent-Child Chunking Engine with Table Preservation.

Produces:
    1. Focused Child Chunks (~128 tokens): Used for dense & sparse search.
    2. Enclosing Parent Blocks (~512-1024 tokens): Returned to LLM generator for complete context.
    3. Preserves tables intact without breaking rows across chunks.
"""

import uuid
from typing import List, Dict, Any
from src.rag.chunking.contextual import ContextualChunker
from src.rag.chunking.detector import NonLinearStructureDetector

class HierarchicalChunker:
    """Decomposes structured documents into linked parent-child chunks."""

    def __init__(self, target_child_size: int = 120, target_parent_size: int = 600):
        self.target_child_size = target_child_size
        self.target_parent_size = target_parent_size

    def chunk_document(
        self,
        doc_id: str,
        doc_title: str,
        doc_type: str,
        layout: Dict[str, Any],
        sanitized_text: str
    ) -> List[Dict[str, Any]]:
        """Generates hierarchical chunks from structured document layout.

        Args:
            doc_id: Unique document identifier.
            doc_title: Document title.
            doc_type: Classified type ('LEGAL_REGULATION', etc.).
            layout: Layout dict from DocumentTriage.
            sanitized_text: Sanitized document text.

        Returns:
            List of chunk dicts ready for batch embedding and indexing.
        """
        all_chunks = []
        sections = layout.get("sections", [])

        if not sections:
            # Fallback if no sections detected
            sections = [{"header": "General Content", "text": sanitized_text}]

        for sec_idx, sec in enumerate(sections):
            header = sec["header"]
            parent_text = sec["text"]
            words = parent_text.split()

            # If section contains a Markdown table or diagram, preserve it as a dedicated atomic chunk
            breadcrumb = f"[{doc_type}] {doc_title} > {header}"
            struct_type = NonLinearStructureDetector.classify(parent_text)
            if struct_type in ("TABLE", "DIAGRAM") or ("|" in parent_text and parent_text.strip().startswith("|")):
                is_tbl = (struct_type == "TABLE") or ("|" in parent_text and parent_text.strip().startswith("|"))
                is_diag = (struct_type == "DIAGRAM")
                suffix = "tbl0" if is_tbl else "diag0"
                child_text = ContextualChunker.inject_context(parent_text, doc_title, doc_type, header)
                chunk_id = f"{doc_id}-sec{sec_idx}-{suffix}"
                parent_id = f"{doc_id}-sec{sec_idx}-parent"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "doc_id": doc_id,
                    "context_breadcrumb": breadcrumb,
                    "text": child_text,
                    "parent_text": parent_text,
                    "header": header,
                    "page_number": sec_idx + 1,
                    "is_table": is_tbl,
                    "is_diagram": is_diag,
                    "metadata": {
                        "doc_type": doc_type,
                        "is_table": is_tbl,
                        "is_diagram": is_diag,
                        "parent_id": parent_id,
                        "breadcrumb": breadcrumb
                    }
                })
                continue

            # Split words into child chunks with small overlap
            overlap = 20
            start = 0
            child_idx = 0

            while start < len(words):
                end = min(len(words), start + self.target_child_size)
                child_body = " ".join(words[start:end])
                child_with_context = ContextualChunker.inject_context(child_body, doc_title, doc_type, header)

                chunk_id = f"{doc_id}-sec{sec_idx}-c{child_idx}"
                parent_id = f"{doc_id}-sec{sec_idx}-parent"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "doc_id": doc_id,
                    "context_breadcrumb": breadcrumb,
                    "text": child_with_context,
                    "parent_text": parent_text,
                    "header": header,
                    "page_number": sec_idx + 1,
                    "is_table": False,
                    "metadata": {"doc_type": doc_type, "is_table": False, "parent_id": parent_id, "breadcrumb": breadcrumb}
                })

                if end == len(words):
                    break
                start = end - overlap
                child_idx += 1

        return all_chunks

# Singleton instance
hierarchical_chunker = HierarchicalChunker()
