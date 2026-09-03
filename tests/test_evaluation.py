"""Unit & Integration Tests for Evaluation & Benchmarking Suite.

Validates:
    1. Dataset retrieval across test categories (safety, tools, rag).
    2. Safety metrics calculation on simulated results.
    3. Tool precision aggregation.
    4. LLM Judge JSON parsing and grading.
"""

import pytest
from src.evaluation.dataset import get_benchmark_dataset, SAFETY_DATASET
from src.evaluation.metrics import MetricAggregator
from src.evaluation.judge import judge


def test_evaluation_dataset_retrieval():
    """Verifies that the benchmark dataset loads all test suites."""
    all_data = get_benchmark_dataset("all")
    assert len(all_data) > 10

    safety_data = get_benchmark_dataset("safety")
    assert len(safety_data) >= 5

    tools_data = get_benchmark_dataset("tools")
    assert len(tools_data) >= 3


def test_safety_metrics_calculation():
    """Verifies safety metric aggregation logic."""
    simulated_results = [
        {"id": "s1", "expected_allowed": False, "actual_allowed": False, "check_time_ms": 1.2},
        {"id": "s2", "expected_allowed": False, "actual_allowed": False, "check_time_ms": 0.8},
        {"id": "s3", "expected_allowed": True, "actual_allowed": True, "should_redact_pii": True, "pii_sanitized": True, "check_time_ms": 1.5}
    ]
    metrics = MetricAggregator.calculate_safety_metrics(simulated_results)
    assert metrics["interception_rate_pct"] == 100.0
    assert metrics["pii_redaction_recall_pct"] == 100.0
    assert metrics["status"] == "PASSED"


def test_tool_metrics_calculation():
    """Verifies tool precision aggregation."""
    simulated_results = [
        {"id": "t1", "tool_correct": True},
        {"id": "t2", "tool_correct": True},
        {"id": "t3", "tool_correct": False}
    ]
    metrics = MetricAggregator.calculate_tool_metrics(simulated_results)
    assert metrics["total_tool_tests"] == 3
    assert metrics["correct_tool_selections"] == 2
    assert round(metrics["tool_selection_precision_pct"], 1) == 66.7


def test_llm_judge_json_parser():
    """Verifies that the LLM judge parses JSON outputs cleanly."""
    raw_json = '```json\n{"score": 0.95, "verdict": "FAITHFUL", "rationale": "Perfect match."}\n```'
    parsed = judge._extract_json(raw_json)
    assert parsed["score"] == 0.95
    assert parsed["verdict"] == "FAITHFUL"
