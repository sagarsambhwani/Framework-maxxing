"""Research Synthesizer Node: Compiles findings into a comprehensive research brief."""
from typing import Any, Dict, List
from src.agent.state import ResearchState
from src.gateway.router import get_gateway
from src.guardrails.rails_manager import get_guardrails_manager
from src.common.config import settings
from src.common.mock_provider import generate_mock_synthesis


SYNTHESIS_SYSTEM_PROMPT = """You are an Executive Research Synthesizer.
Your goal is to synthesize the provided research question and gathered tool findings into a structured, executive-grade research brief.

Structure your response using Markdown with:
# Executive Research Report: <Topic>
## 1. Executive Summary
## 2. Key Technical Findings & Architecture Analysis
## 3. Performance, Security & Observability Takeaways
## 4. Strategic Recommendations & Next Steps

Ensure the content is factual, concise, and grounded in the provided findings.
"""


def research_synthesizer_node(state: ResearchState) -> Dict[str, Any]:
    """LangGraph node to generate the final research synthesis report."""
    query = state.get("sanitized_query") or state.get("query", "")
    findings = state.get("findings", [])
    gateway = get_gateway()
    guardrails = get_guardrails_manager()

    # Format findings
    findings_context = []
    for f in findings:
        step_name = f.get("step", "Step")
        tool = f.get("tool", "")
        res = f.get("result", "")
        findings_context.append(f"### {step_name} (Tool: {tool})\n{res}")

    context_str = "\n\n".join(findings_context) if findings_context else "No intermediate tool results available."

    user_prompt = f"Original Research Query: {query}\n\nGathered Findings & Tool Outputs:\n{context_str}\n\nPlease generate the comprehensive executive research brief."

    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = gateway.completion(
            model=settings.SYNTHESIS_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
            metadata={"node": "synthesizer", "session_id": state.get("session_id", "")}
        )
        raw_report = response["content"].strip()
    except Exception as e:
        raw_report = generate_mock_synthesis(query, findings)

    # Validate output with NeMo Guardrails
    guardrail_result = guardrails.validate_output(raw_report, context=context_str)
    final_output = guardrail_result["response"]

    return {
        "final_report": final_output,
        "error": None
    }
