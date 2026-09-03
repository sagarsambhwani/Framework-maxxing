"""Unit & Integration Tests for Agentic Marketing Workflow & Evaluation Suite.

Validates:
    1. Marketing tools (readability analyzer, keyword density).
    2. End-to-end marketing LangGraph workflow (Researcher -> Strategist -> Copywriter -> Critic -> Publisher).
    3. Self-correction reflection loop functionality.
    4. LLM Judge marketing copy evaluation.
"""

import pytest
from src.workflows.marketing.tools import readability_analyzer, keyword_density_analyzer
from src.workflows.marketing.graph import marketing_workflow
from src.evaluation.judge import judge


def test_marketing_readability_analyzer():
    """Verifies that the readability tool accurately computes character and word counts."""
    text = "HyperGateway AI reduces latency by 70% with 0ms caching."
    metrics = readability_analyzer(text)
    assert metrics["char_count"] == len(text)
    assert metrics["word_count"] > 5
    assert metrics["est_reading_time_sec"] > 0


def test_marketing_keyword_density():
    """Verifies keyword occurrence and density calculation."""
    text = "AI Gateway is the future. An AI Gateway saves money."
    stats = keyword_density_analyzer(text, ["AI Gateway", "latency"])
    assert stats["keywords"]["AI Gateway"]["count"] == 2
    assert stats["keywords"]["latency"]["count"] == 0


def test_marketing_workflow_execution():
    """Verifies end-to-end multi-agent marketing campaign generation and approval."""
    initial_state = {
        "brief": "Launch an open-source voice AI tool with sub-200ms transcription.",
        "product_name": "WhisperFlow",
        "target_audience": "Developers",
        "brand_voice": "Energetic & Direct",
        "target_channels": ["twitter", "linkedin", "email"],
        "session_id": "test-mkt-session",
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "research_insights": [],
        "campaign_angles": [],
        "copy_drafts": {},
        "critic_feedback": [],
        "critic_approved": False,
        "revision_count": 0,
        "final_campaign": {}
    }

    final_state = marketing_workflow.invoke(initial_state)
    assert final_state is not None
    assert final_state.get("guardrail_allowed") is True
    assert len(final_state.get("research_insights", [])) > 0
    assert len(final_state.get("campaign_angles", [])) > 0

    campaign = final_state.get("final_campaign", {})
    assert campaign.get("status") == "APPROVED"
    assert "assets" in campaign
    assert "twitter_x" in campaign["assets"]
    assert "linkedin" in campaign["assets"]


def test_llm_judge_marketing_copy():
    """Verifies that the LLM copy judge scores brand voice and CTA strength."""
    brief = "AI Gateway for low latency"
    target_voice = "Technical & Precise"
    copy_assets = {
        "twitter": "Cut LLM latency by 70% with HyperGateway #AI",
        "email": "Subject: Cut your AI latency\n\nExperience sub-second inference."
    }

    evaluation = judge.evaluate_marketing_copy(brief, target_voice, copy_assets)
    assert evaluation is not None
    assert "voice_score" in evaluation
    assert 0.0 <= evaluation["voice_score"] <= 1.0
    assert "hook_score" in evaluation
    assert "cta_score" in evaluation
