"""05_evaluation_benchmark.py - Enterprise AI Evaluation & Benchmarking Suite Demo.

Demonstrates:
    1. Red-teaming security evaluation on 7 adversarial attack vectors.
    2. Tool selection accuracy & math precision evaluation.
    3. RAG Triad Faithfulness & Answer Relevance using LLM-as-a-Judge.
    4. Performance benchmarking (Groq LPU vs 0ms memory cache).
    5. Automatic Markdown Scorecard export & Langfuse Cloud sync.

Run with:
    .venv\\Scripts\\python.exe examples/05_evaluation_benchmark.py
"""

import sys
import os

# Add root directory to pythonpath
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.runner import run_evaluation_suite

if __name__ == "__main__":
    export_file = "evaluation_report.md"
    summary = run_evaluation_suite(export_path=export_file)
    print(f"\n✓ Evaluation complete! Full report saved to: {export_file}")
