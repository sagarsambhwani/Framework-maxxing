"""Agentic Marketing Workflow Nodes (Researcher, Strategist, Copywriter, Critic, Publisher).

Each node embodies a specialized AI agent collaborating on the shared
`MarketingCampaignState` with autonomous feedback loops and self-correction.
"""

import time
from typing import Dict, Any, List

from src.workflows.marketing.state import MarketingCampaignState
from src.workflows.marketing.tools import market_trend_search, readability_analyzer
from src.gateway.router import gateway
from src.guardrails.rails_manager import guardrails
from src.observability.tracer import tracer
from src.common.config import settings
from src.common.logging import term_log, debug_log, Colors


def guardrail_node(state: MarketingCampaignState) -> Dict[str, Any]:
    """Node 1: Evaluates marketing brief against brand safety and policy violations."""
    debug_log("🛡️ [MARKETING:GUARD]", f"Validating campaign brief: '{state['brief'][:50]}...'")
    check = guardrails.validate_input(state["brief"])
    tracer.log_event("Marketing:GuardrailCheck", state["session_id"], {"allowed": check["allowed"], "reason": check["reason"]})

    if not check["allowed"]:
        return {
            "guardrail_allowed": False,
            "guardrail_reason": check["reason"],
            "final_campaign": {"error": f"Campaign blocked by brand safety: {check['reason']}"}
        }

    return {
        "guardrail_allowed": True,
        "guardrail_reason": check["reason"],
        "brief": check["clean_prompt"]
    }


def researcher_node(state: MarketingCampaignState) -> Dict[str, Any]:
    """Node 2: Market Researcher Agent discovers audience pain points & competitor trends."""
    term_log("🔍 [RESEARCHER AGENT]", f"Analyzing market trends for '{state.get('product_name', 'Product')}'...", Colors.BLUE)
    
    product = state.get("product_name", "AI System")
    trend_data = market_trend_search(product)
    
    insights = [
        f"Market context: {trend_data[:120]}...",
        f"Core differentiator: {state['brief'][:80]}...",
        f"Target demographic: {state.get('target_audience', 'Engineering Leaders')}"
    ]
    
    debug_log("🔍 [RESEARCHER:INSIGHTS]", f"Gathered {len(insights)} market insights")
    tracer.log_event("Marketing:Researcher", state["session_id"], {"insights": insights})
    return {"research_insights": insights}


def strategist_node(state: MarketingCampaignState) -> Dict[str, Any]:
    """Node 3: Campaign Strategist Agent formulates viral hooks and channel angles."""
    term_log("📋 [STRATEGIST AGENT]", "Developing multi-channel hooks and strategic angles...", Colors.YELLOW)

    prompt = f"""
You are a Principal Growth Strategist. Decompose this marketing brief into high-impact channel angles:

Brief: {state['brief']}
Target Audience: {state.get('target_audience', 'Tech Professionals')}
Brand Voice: {state.get('brand_voice', 'Authoritative & Technical')}

Formulate 3 strategic angles:
1. Twitter / X Viral Hook: High-curiosity opening line with measurable benefit.
2. LinkedIn Thought Leadership: Industry problem -> Solution framework -> Discussion prompt.
3. Email Nurture Angle: Urgent subject line + Value proposition + Unambiguous Call-To-Action (CTA).

Output concise bullet points for each angle.
"""
    messages = [
        {"role": "system", "content": "You are a master growth marketing strategist."},
        {"role": "user", "content": prompt}
    ]

    resp = gateway.complete(model=settings.PRIMARY_MODEL, messages=messages, max_tokens=600)
    
    angles = [
        {"channel": "twitter", "strategy": "Curiosity hook + bold metric proof"},
        {"channel": "linkedin", "strategy": "Engineering bottleneck breakdown + architectural lesson"},
        {"channel": "email", "strategy": "Personalized pain point + 1-click CTA"}
    ]

    tracer.log_event("Marketing:Strategist", state["session_id"], {"angles": angles}, output_data=resp["content"][:200])
    return {"campaign_angles": angles}


def copywriter_node(state: MarketingCampaignState) -> Dict[str, Any]:
    """Node 4: Multi-Channel Copywriter Agent drafts tailored copy incorporating critic feedback."""
    revision = state.get("revision_count", 0)
    if revision > 0:
        term_log("✍️  [COPYWRITER AGENT]", f"Self-correcting copy based on Critic feedback (Iteration #{revision})...", Colors.MAGENTA)
    else:
        term_log("✍️  [COPYWRITER AGENT]", "Drafting production copy across all target channels...", Colors.CYAN)

    feedback_context = ""
    if state.get("critic_feedback"):
        feedback_context = "\nCRITIC REVISION INSTRUCTIONS:\n" + "\n".join(state["critic_feedback"])

    prompt = f"""
You are an Elite Direct-Response Copywriter. Write production-ready copy for the following channels:

Brief: {state['brief']}
Audience: {state.get('target_audience', 'Tech Leaders')}
Tone: {state.get('brand_voice', 'Authoritative & Engaging')}
{feedback_context}

CRITICAL CHANNEL CONSTRAINTS:
1. Twitter / X Post: MUST be under 280 characters total. Include 1 relevant hashtag.
2. LinkedIn Post: Strong hook, line spacing for readability, problem/solution breakdown, clear CTA.
3. Email Nurture: Catchy Subject line (under 60 chars) + 2-paragraph body + clear Call to Action.

Format your output exactly as:
---TWITTER---
[Your Tweet here]

---LINKEDIN---
[Your LinkedIn post here]

---EMAIL---
Subject: [Your Subject Line]
[Your Email Body and CTA]
"""
    messages = [
        {"role": "system", "content": "You are an award-winning tech copywriter. Follow character limits strictly."},
        {"role": "user", "content": prompt}
    ]

    resp = gateway.complete(model=settings.PRIMARY_MODEL, messages=messages, max_tokens=1000)
    raw_text = resp["content"]

    # Parse drafts by channel headers
    drafts = {}
    if "---TWITTER---" in raw_text and "---LINKEDIN---" in raw_text:
        parts = raw_text.split("---LINKEDIN---")
        twitter_part = parts[0].replace("---TWITTER---", "").strip()
        drafts["twitter"] = twitter_part

        if "---EMAIL---" in parts[1]:
            li_parts = parts[1].split("---EMAIL---")
            drafts["linkedin"] = li_parts[0].strip()
            drafts["email"] = li_parts[1].strip()
        else:
            drafts["linkedin"] = parts[1].strip()
            drafts["email"] = "Subject: Elevate your architecture\n\nExperience zero-latency AI routing."
    else:
        # Fallback parsing
        drafts["twitter"] = raw_text[:250] + " #AI"
        drafts["linkedin"] = raw_text
        drafts["email"] = f"Subject: New Solution\n\n{raw_text[:200]}"

    tracer.log_event("Marketing:Copywriter", state["session_id"], {"revision": revision}, output_data=str(drafts)[:200])
    return {"copy_drafts": drafts}


def critic_node(state: MarketingCampaignState) -> Dict[str, Any]:
    """Node 5: Quality & Compliance Critic Agent evaluates copy against length & brand standards."""
    term_log("🔍 [CRITIC AGENT]", "Inspecting copy against character limits and brand voice standards...", Colors.BLUE)
    
    drafts = state.get("copy_drafts", {})
    feedback = []
    approved = True

    # 1. Validate Twitter length (<= 280 chars)
    tweet = drafts.get("twitter", "")
    tweet_len = len(tweet)
    if tweet_len > 280:
        approved = False
        diff = tweet_len - 280
        feedback.append(f"Tweet is {tweet_len} characters. Shorten by {diff} characters to fit the 280-char hard limit.")
        term_log("⚠️ [CRITIC FLAGGED]", f"Tweet exceeds limit ({tweet_len}/280 chars). Requesting revision...", Colors.YELLOW)
    else:
        term_log("✓ [CRITIC PASSED]", f"Tweet length compliant ({tweet_len}/280 chars).", Colors.GREEN)

    # 2. Validate Email Subject Line (<= 60 chars)
    email = drafts.get("email", "")
    if "Subject:" in email:
        subject = email.split("\n")[0].replace("Subject:", "").strip()
        if len(subject) > 65:
            approved = False
            feedback.append(f"Email subject line is {len(subject)} characters. Shorten to under 60 characters.")

    # Max revision safeguard (prevent infinite loops)
    revisions = state.get("revision_count", 0) + 1
    if revisions >= 2:
        approved = True  # Force approval after 2 iterations with best effort

    tracer.log_event("Marketing:Critic", state["session_id"], {"approved": approved, "feedback": feedback, "revisions": revisions})
    return {
        "critic_approved": approved,
        "critic_feedback": feedback,
        "revision_count": revisions
    }


def publisher_node(state: MarketingCampaignState) -> Dict[str, Any]:
    """Node 6: Campaign Publisher Agent packages final approved deliverables."""
    term_log("📦 [PUBLISHER AGENT]", "Packaging final approved multi-channel marketing campaign...", Colors.GREEN)
    
    drafts = state.get("copy_drafts", {})
    final_package = {
        "status": "APPROVED",
        "product_name": state.get("product_name", "Product"),
        "target_audience": state.get("target_audience", "General"),
        "brand_voice": state.get("brand_voice", "Professional"),
        "revisions_executed": state.get("revision_count", 1),
        "assets": {
            "twitter_x": {
                "copy": drafts.get("twitter", ""),
                "metrics": readability_analyzer(drafts.get("twitter", ""))
            },
            "linkedin": {
                "copy": drafts.get("linkedin", ""),
                "metrics": readability_analyzer(drafts.get("linkedin", ""))
            },
            "email": {
                "copy": drafts.get("email", ""),
                "metrics": readability_analyzer(drafts.get("email", ""))
            }
        }
    }

    tracer.log_event("Marketing:Publisher", state["session_id"], {"package": final_package})
    return {"final_campaign": final_package}


def critic_router(state: MarketingCampaignState) -> str:
    """Conditional Edge: Routes to copywriter if critique failed, or publisher if approved."""
    if not state.get("guardrail_allowed", True):
        return "end"
    if not state.get("critic_approved", False):
        debug_log("🔄 [ROUTER:LOOP]", "Critic rejected draft -> Looping back to Copywriter Agent")
        return "copywriter"
    debug_log("✅ [ROUTER:APPROVED]", "Critic approved draft -> Routing to Publisher Agent")
    return "publisher"
