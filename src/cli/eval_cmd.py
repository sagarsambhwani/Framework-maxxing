"""Evaluation CLI Command Handler."""

def run_evaluation(export_path: str = "evaluation_report.md"):
    """Executes the enterprise evaluation & benchmarking suite."""
    from src.evaluation.runner import run_evaluation_suite
    run_evaluation_suite(export_path=export_path)
