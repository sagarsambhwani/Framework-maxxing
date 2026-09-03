"""06_marketing_agent_workflow.py - Autonomous Multi-Agent Marketing Campaign Generator.

Demonstrates:
    1. Researcher Agent: Analyzes live market trends and competitor keywords.
    2. Strategist Agent: Formulates viral hooks and channel strategies.
    3. Copywriter Agent: Generates production-ready copy for Twitter/X, LinkedIn, and Email.
    4. Critic Agent: Enforces 280-char Twitter limit, readability, and brand voice standards.
    5. Reflection & Self-Correction Loop: Automatically iterates if copy fails quality constraints.
    6. Publisher Agent: Outputs structured JSON campaign deliverables.

Run with:
    .venv\\Scripts\\python.exe examples/06_marketing_agent_workflow.py
"""

import sys
import os
import json
import uuid
import time

# Add root directory to pythonpath
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workflows.marketing.graph import marketing_workflow
from src.common.logging import print_banner, Colors

if __name__ == "__main__":
    brief = "Launch an ultra-fast AI Gateway that reduces inference costs by 70% with 0ms in-memory caching for enterprise teams."
    product_name = "HyperGateway AI"
    audience = "Enterprise CTOs and Lead AI Engineers"
    brand_voice = "Authoritative, Precise & High-Energy"
    channels = ["twitter", "linkedin", "email"]
    session_id = f"mkt-{uuid.uuid4().hex[:6]}"

    print_banner(
        "AUTONOMOUS AGENTIC MARKETING WORKFLOW",
        f"Product: {product_name} | Channels: {', '.join(channels)}"
    )

    t0 = time.time()
    final_state = marketing_workflow.invoke({
        "brief": brief,
        "product_name": product_name,
        "target_audience": audience,
        "brand_voice": brand_voice,
        "target_channels": channels,
        "session_id": session_id,
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "research_insights": [],
        "campaign_angles": [],
        "copy_drafts": {},
        "critic_feedback": [],
        "critic_approved": False,
        "revision_count": 0,
        "final_campaign": {}
    })
    dur = round(time.time() - t0, 2)

    campaign = final_state.get("final_campaign", {})
    assets = campaign.get("assets", {})

    print("\n" + "=" * 80)
    print("📣 APPROVED MULTI-CHANNEL CAMPAIGN DELIVERABLES")
    print("=" * 80)
    print(f"Status           : {Colors.GREEN}{campaign.get('status', 'APPROVED')}{Colors.END}")
    print(f"Target Audience  : {campaign.get('target_audience')}")
    print(f"Brand Voice      : {campaign.get('brand_voice')}")
    print(f"Reflection Loops : {campaign.get('revisions_executed', 1)} iteration(s)")
    print("-" * 80)

    # 1. Twitter Post
    tw = assets.get("twitter_x", {})
    print(f"\n🐦 [TWITTER / X POST] ({tw.get('metrics', {}).get('char_count', 0)}/280 chars):")
    print(f"{tw.get('copy')}\n")

    # 2. LinkedIn Post
    li = assets.get("linkedin", {})
    print(f"💼 [LINKEDIN THOUGHT LEADERSHIP] ({li.get('metrics', {}).get('word_count', 0)} words):")
    print(f"{li.get('copy')}\n")

    # 3. Email Nurture
    em = assets.get("email", {})
    print(f"📧 [EMAIL NURTURE]:")
    print(f"{em.get('copy')}\n")

    print("=" * 80)
    print(f"⏱️  Campaign generated and validated in {dur}s | Traces synced to Langfuse")
    print("=" * 80 + "\n")
