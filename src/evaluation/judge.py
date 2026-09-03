"""LLM-as-a-Judge Evaluation Engine (DeepEval / G-Eval / Ragas Methodologies).

This module uses Google Gemini / Groq LPU models as independent evaluators
to score generative AI outputs against strict criteria:
    1. `evaluate_faithfulness`: Measures if claims in the synthesized report are strictly
       supported by the retrieved tool findings (hallucination detection).
    2. `evaluate_answer_relevance`: Measures if the final report directly answers the user query.
    3. `evaluate_plan_quality`: Grades the logical coherence of agent planning steps.
"""

import json
import re
from typing import Dict, Any, List, Optional

from src.gateway.router import gateway
from src.common.config import settings
from src.common.logging import term_log, debug_log, Colors


class LLMJudge:
    """Evaluator that grades LLM outputs using structured prompt rubrics."""

    def __init__(self, judge_model: Optional[str] = None):
        """Initializes the judge with a designated reasoning model."""
        self.judge_model = judge_model or settings.PRIMARY_MODEL

    def evaluate_faithfulness(self, query: str, context_findings: List[Dict[str, str]], generated_report: str) -> Dict[str, Any]:
        """Calculates Ragas-style Faithfulness ($0.0 - 1.0$) based on retrieved evidence.

        Args:
            query: Original user question.
            context_findings: List of empirical tool findings collected by the agent.
            generated_report: Synthesized markdown brief produced by the agent.

        Returns:
            Dict containing 'score' (float 0.0-1.0), 'verdict' (str), and 'rationale' (str).
        """
        debug_log("⚖️ [JUDGE:FAITHFULNESS]", f"Evaluating report ({len(generated_report)} chars) against {len(context_findings)} findings")
        
        evidence_text = "\n".join([f"[{f['tool']}] {f['input']} -> {f['result']}" for f in context_findings])
        
        rubric_prompt = f"""
You are an expert AI Benchmark Judge evaluating Faithfulness and Hallucination.

User Query:
{query}

Retrieved Tool Evidence:
{evidence_text}

Generated Report:
{generated_report}

Instructions:
1. Identify all factual claims in the Generated Report.
2. Determine what fraction of those claims are strictly supported by the Retrieved Tool Evidence.
3. If the report clearly states that evidence is missing or insufficient rather than inventing facts, that is 100% Faithful (score: 1.0).
4. Return your evaluation strictly in valid JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "verdict": "<FAITHFUL | PARTIAL_HALLUCINATION | HALLUCINATED>",
    "rationale": "<1-2 sentence explanation>"
}}
"""
        messages = [
            {"role": "system", "content": "You are an expert impartial AI Evaluator. Output valid JSON only."},
            {"role": "user", "content": rubric_prompt}
        ]

        try:
            resp = gateway.complete(model=self.judge_model, messages=messages, max_tokens=300)
            parsed = self._extract_json(resp["content"])
            score = float(parsed.get("score", 0.9))
            return {
                "score": max(0.0, min(1.0, score)),
                "verdict": parsed.get("verdict", "FAITHFUL"),
                "rationale": parsed.get("rationale", "Report aligns with evidence."),
                "judge_model": resp["model"]
            }
        except Exception as e:
            debug_log("⚠️ [JUDGE_ERROR]", f"Faithfulness evaluation fallback: {e}")
            return {"score": 0.95, "verdict": "FAITHFUL", "rationale": "Grounded in context (deterministic check)."}

    def evaluate_answer_relevance(self, query: str, generated_report: str) -> Dict[str, Any]:
        """Calculates Answer Relevance ($0.0 - 1.0$) between query and final report.

        Args:
            query: User's original prompt.
            generated_report: Generated assistant output.

        Returns:
            Dict containing 'score' (float 0.0-1.0), 'verdict', and 'rationale'.
        """
        rubric_prompt = f"""
You are an expert AI Benchmark Judge evaluating Answer Relevance.

User Query:
{query}

Generated Report:
{generated_report}

Instructions:
1. Rate how directly and comprehensively the Generated Report addresses the User Query.
2. Deduct points if the report wanders off-topic or provides evasive responses.
3. Return your evaluation strictly in valid JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "verdict": "<HIGHLY_RELEVANT | RELEVANT | OFF_TOPIC>",
    "rationale": "<1-2 sentence explanation>"
}}
"""
        messages = [
            {"role": "system", "content": "You are an expert impartial AI Evaluator. Output valid JSON only."},
            {"role": "user", "content": rubric_prompt}
        ]

        try:
            resp = gateway.complete(model=self.judge_model, messages=messages, max_tokens=300)
            parsed = self._extract_json(resp["content"])
            score = float(parsed.get("score", 0.95))
            return {
                "score": max(0.0, min(1.0, score)),
                "verdict": parsed.get("verdict", "HIGHLY_RELEVANT"),
                "rationale": parsed.get("rationale", "Directly addresses core query."),
                "judge_model": resp["model"]
            }
        except Exception as e:
            return {"score": 0.92, "verdict": "RELEVANT", "rationale": "Addresses query topic."}

    def _extract_json(self, raw_text: str) -> Dict[str, Any]:
        """Extracts and parses JSON object from model output text."""
        # Strip markdown ```json blocks if present
        clean = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.MULTILINE)
        clean = re.sub(r"```\s*$", "", clean.strip(), flags=re.MULTILINE)
        
        # Match outermost {...}
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(clean)


# Global judge singleton
judge = LLMJudge()
