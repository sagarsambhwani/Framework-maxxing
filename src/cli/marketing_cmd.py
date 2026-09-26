"""Marketing Campaign CLI Command Handler."""

import time
import uuid
from src.common.logging import print_banner, Colors


def run_marketing(brief: str):
    """Executes the Agentic Marketing Campaign Workflow."""
    from src.workflows.marketing.graph import marketing_workflow

    session_id = f"mkt-{uuid.uuid4().hex[:6]}"

    print_banner("AGENTIC MARKETING WORKFLOW", f"Brief: '{brief[:60]}...' | Session: {session_id}")
    start_t = time.time()

    final_state = marketing_workflow.invoke({
        "brief": brief,
        "product_name": "New AI Product",
        "target_audience": "Tech Leaders & Engineers",
        "brand_voice": "Authoritative, Direct & Engaging",
        "target_channels": ["twitter", "linkedin", "email"],
        "session_id": session_id,
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "research_insights": [],
        "campaign_angles": [],
        "copy_drafts": {},
        "critic_feedback": [],
        "critic_approved": False,
        "revision_count": 0,
        "final_campaign_report": ""
    })
    dur = round(time.time() - start_t, 2)

    print("\n" + "=" * 80)
    print("📊 MARKETING WORKFLOW RESULTS")
    print("=" * 80)
    print(f"⏱️  Duration         : {dur}s")
    print(f"📈 Revisions Run    : {final_state.get('revision_count', 0)}")
    print(f"🔍 Research Insights: {len(final_state.get('research_insights', []))} points gathered")
    print(f"💡 Strategic Angles : {len(final_state.get('campaign_angles', []))} angles conceived")
    print(f"✍️  Channels Drafted : {list(final_state.get('copy_drafts', {}).keys())}")
    print(f"\n{Colors.GREEN}================ CAMPAIGN DELIVERABLES ================{Colors.END}")
    print(final_state.get("final_campaign_report", ""))
    print("=" * 80 + "\n")
