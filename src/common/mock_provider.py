"""Mock Provider for Offline Simulation & Testing."""
import json
import time
from typing import Any, Dict, List, Optional


class MockResponse:
    def __init__(self, content: str, model: str = "mock-simulated-model"):
        self.content = content
        self.model = model
        self.usage = {
            "prompt_tokens": len(content.split()) * 2,
            "completion_tokens": len(content.split()),
            "total_tokens": len(content.split()) * 3,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": f"mock-{int(time.time())}",
            "choices": [{"message": {"role": "assistant", "content": self.content}}],
            "model": self.model,
            "usage": self.usage,
        }


def generate_mock_plan(query: str) -> str:
    return json.dumps({
        "thought": f"Decomposing research query: '{query}' into structured analytical tasks.",
        "steps": [
            {
                "step_id": 1,
                "description": f"Perform market and technical search on {query}",
                "tool": "web_search",
                "tool_input": f"{query} latest developments architecture benchmark"
            },
            {
                "step_id": 2,
                "description": "Calculate estimated throughput and latency benchmarks",
                "tool": "calculator",
                "tool_input": "1000 * 150 / 60"
            },
            {
                "step_id": 3,
                "description": "Summarize key architectural tradeoffs and takeaways",
                "tool": "summarizer",
                "tool_input": "Synthesizing research findings into executive brief"
            }
        ]
    }, indent=2)


def generate_mock_synthesis(query: str, findings: List[Dict[str, Any]]) -> str:
    findings_str = "\n".join([f"- **{f.get('step', 'Step')}**: {f.get('result', '')[:200]}" for f in findings])
    return f"""# Executive Research Report: {query}

## 1. Executive Summary
This report analyzes the architecture, trade-offs, and operational telemetry for **{query}**. 
Based on empirical data and system benchmarks, the integrated gateway and agentic routing layer provides high resiliency, sub-second latency routing, and end-to-end security compliance.

## 2. Key Findings & Empirical Data
{findings_str}

## 3. Architecture & Operational Recommendations
1. **Multi-Model Routing**: Implement dynamic fallback chains to mitigate rate limits and provider outages.
2. **Layered Guardrails**: Enforce deterministic Colang input/output boundary policies to prevent prompt injection and unauthorized jailbreaks.
3. **Continuous Observability**: Track token consumption, TTFT (Time to First Token), and cost metrics per session in Langfuse.

## 4. Conclusion
The proposed architecture meets production requirements for scalability, governance, and autonomous problem decomposition.
"""
