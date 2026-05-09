"""Build the ideation agent graph.

Flow: START → init → agent_loop → generate_report → extract_ideas → approve_ideas → END
                                                                         ↓ (feedback)
                                                                     agent_loop
"""

from langgraph.graph import END, START, StateGraph
from loguru import logger

from scider.core.approval import make_approval_node
from scider.core.types import Message

from . import execute
from .state import IdeationAgentState


def init_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    logger.debug("init_node of IdeationAgent")

    content = f"Generate novel research ideas for: {agent_state.user_query}"
    if agent_state.research_domain:
        content += f"\n\nResearch domain: {agent_state.research_domain}"

    agent_state.add_message(
        Message(role="user", content=content, agent_sender="ideation").with_log()
    )

    agent_state.intermediate_state.append({"node_name": "init", "output": "Starting ideation."})
    return agent_state


def _format_ideas_summary(agent_state: IdeationAgentState) -> str:
    """Format research ideas for approval display."""
    if not agent_state.research_ideas:
        return agent_state.output_summary or "No ideas generated."

    lines = ["# Research Ideas\n"]
    for i, idea in enumerate(agent_state.research_ideas, 1):
        title = idea.get("title", "Untitled")
        desc = idea.get("description", "")
        composite = idea.get("composite_score")
        novelty = idea.get("novelty_score")
        if composite is not None:
            score_str = f" (composite: {composite:.2f})"
        elif novelty is not None:
            score_str = f" (novelty: {novelty}/10)"
        else:
            score_str = ""
        operator = idea.get("search_operator")
        if operator and operator != "seed":
            score_str += f" [{operator}]"
        lines.append(f"**{i}. {title}**{score_str}")
        if desc:
            lines.append(f"   {desc}")
        lines.append("")

    has_composite = any(i.get("composite_score") is not None for i in agent_state.research_ideas)
    if has_composite and agent_state.idea_score is not None:
        lines.append(f"**Best composite score: {agent_state.idea_score:.3f}**")
    elif agent_state.idea_score is not None:
        lines.append(f"**Average novelty score: {agent_state.idea_score:.1f}/10**")

    return "\n".join(lines)


approve_ideas_node, approve_ideas_conditional = make_approval_node(
    node_name="approve_ideas",
    summary_extractor=_format_ideas_summary,
    retry_target="agent_loop",
    next_target=END,
    title="Review the generated research ideas",
)


@logger.catch
def build():
    g = StateGraph(IdeationAgentState)

    g.add_node("init", init_node)
    g.add_node("agent_loop", execute.agent_loop_node)
    g.add_node("generate_report", execute.generate_report_node)
    g.add_node("extract_ideas", execute.extract_ideas_node)
    g.add_node("idea_search", execute.idea_search_node)
    g.add_node("approve_ideas", approve_ideas_node)

    g.add_edge(START, "init")
    g.add_edge("init", "agent_loop")
    g.add_edge("agent_loop", "generate_report")
    g.add_edge("generate_report", "extract_ideas")
    g.add_edge("extract_ideas", "idea_search")
    g.add_edge("idea_search", "approve_ideas")
    g.add_conditional_edges("approve_ideas", approve_ideas_conditional)

    return g
