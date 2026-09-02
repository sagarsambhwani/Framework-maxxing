"""Metrics and Telemetry Collection."""
from typing import Any, Dict, List


class MetricsCollector:
    """Aggregates latency, token usage, cost estimation, and tool invocations."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.tool_executions = 0
        self.model_calls: List[Dict[str, Any]] = []

    def record_llm_call(self, model: str, prompt_tokens: int, completion_tokens: int, latency: float):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_latency += latency
        self.model_calls.append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency": round(latency, 3),
        })

    def record_tool_call(self):
        self.tool_executions += 1

    def calculate_estimated_cost(self) -> float:
        """Rough estimation based on average blended token pricing ($0.50/M input, $1.50/M output)."""
        input_cost = (self.total_prompt_tokens / 1_000_000) * 0.50
        output_cost = (self.total_completion_tokens / 1_000_000) * 1.50
        return round(input_cost + output_cost, 6)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_calls": len(self.model_calls),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_latency_seconds": round(self.total_latency, 3),
            "tool_executions": self.tool_executions,
            "estimated_cost_usd": self.calculate_estimated_cost(),
            "model_breakdown": self.model_calls
        }
