"""Non-Linear Structure Detector (Tables, Graphs & Diagrams).

Detects non-linear text formats that require atomic preservation during chunking:
    1. Markdown tables & CSV grids
    2. Diagrams & Flowcharts (Mermaid, Graphviz DOT, PlantUML)
    3. Structural trees (ASCII folder hierarchies)
"""

import re


class NonLinearStructureDetector:
    """Deterministic fast (<1ms) heuristic detector for non-linear structures."""

    # Matches Markdown table divider syntax: e.g. | --- | :---: | ---: |
    TABLE_DIVIDER_REGEX = re.compile(r"^\s*\|?\s*[-:]{2,}\s*\|\s*[-:]{2,}", re.MULTILINE)

    # Matches diagram / flowchart code fences
    DIAGRAM_FENCE_REGEX = re.compile(
        r"```(?:mermaid|dot|graphviz|plantuml)\b[\s\S]*?```",
        re.IGNORECASE
    )

    # ASCII Tree hierarchy characters
    TREE_SYMBOLS = ("├──", "└──", "│   ", "+--", "\\--")

    @classmethod
    def is_markdown_table(cls, text: str) -> bool:
        """Determines if text contains or represents a Markdown table."""
        if "|" in text:
            # Check for standard markdown table divider syntax
            if cls.TABLE_DIVIDER_REGEX.search(text):
                return True
            # Fallback for simple pipe-delimited lines (at least 2 lines with multiple pipes)
            pipe_lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip().startswith("|") and line.strip().endswith("|")
            ]
            if len(pipe_lines) >= 2:
                return True
        return False

    @classmethod
    def is_diagram(cls, text: str) -> bool:
        """Determines if text contains a flowchart, graph, or tree diagram."""
        if cls.DIAGRAM_FENCE_REGEX.search(text):
            return True
        if any(keyword in text for keyword in ("graph TD", "graph LR", "flowchart TD", "flowchart LR", "digraph ")):
            return True
        # Check for ASCII tree structures
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) >= 2:
            tree_matches = sum(1 for line in lines if any(sym in line for sym in cls.TREE_SYMBOLS))
            if tree_matches / len(lines) >= 0.3:
                return True
        return False

    @classmethod
    def classify(cls, text: str) -> str:
        """Classifies the dominant structure of a text block.

        Returns:
            'TABLE', 'DIAGRAM', or 'PROSE'.
        """
        if cls.is_markdown_table(text):
            return "TABLE"
        if cls.is_diagram(text):
            return "DIAGRAM"
        return "PROSE"
