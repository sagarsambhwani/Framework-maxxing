r"""LangGraph Stateful Marketing Campaign Workflow Definition.

Assembles the multi-agent graph with reflection, self-correction loops, and brand quality gates.

Graph Topology:
    START ---> [guardrail_node]
                     |
            (is brief safe?)
            /              \
        [Safe]           [Blocked]
          |                  |
    [researcher_node]       END (Terminated)
          |
    [strategist_node]
          |
    [copywriter_node] <---------------+
          |                           |
     [critic_node]                    |
          |                           |
   (critic approved?)                 |
       /         \                    |
   [Approved]  [Needs Revision]-------+
       |
  [publisher_node]
       |
      END (Campaign Published)
"""

from langgraph.graph import StateGraph, START, END

from src.workflows.marketing.state import MarketingCampaignState
from src.workflows.marketing.nodes import (
    guardrail_node,
    researcher_node,
    strategist_node,
    copywriter_node,
    critic_node,
    publisher_node,
    critic_router
)


def guardrail_branch(state: MarketingCampaignState) -> str:
    """Conditional Edge from Guardrail node."""
    return "researcher" if state.get("guardrail_allowed", True) else "end"


# Define StateGraph
marketing_graph = StateGraph(MarketingCampaignState)

# Add Agent Nodes
marketing_graph.add_node("guardrail", guardrail_node)
marketing_graph.add_node("researcher", researcher_node)
marketing_graph.add_node("strategist", strategist_node)
marketing_graph.add_node("copywriter", copywriter_node)
marketing_graph.add_node("critic", critic_node)
marketing_graph.add_node("publisher", publisher_node)

# Add Edges & Conditional Routing Loops
marketing_graph.add_edge(START, "guardrail")
marketing_graph.add_conditional_edges("guardrail", guardrail_branch, {"researcher": "researcher", "end": END})
marketing_graph.add_edge("researcher", "strategist")
marketing_graph.add_edge("strategist", "copywriter")
marketing_graph.add_edge("copywriter", "critic")
marketing_graph.add_conditional_edges("critic", critic_router, {"copywriter": "copywriter", "publisher": "publisher", "end": END})
marketing_graph.add_edge("publisher", END)

# Compile Marketing Workflow Execution Artifact
marketing_workflow = marketing_graph.compile()
