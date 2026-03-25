from langgraph.graph import END, START, StateGraph
from loguru import logger

from scider.core.approval import make_approval_node, make_selection_approval_node

from . import execute
from .state import IdeationAgentState


def _format_ideas_summary(state: IdeationAgentState) -> str:
    """Format research ideas for user review."""
    if not state.research_ideas:
        return "No research ideas generated."
    lines = [f"Generated {len(state.research_ideas)} research ideas:\n"]
    for i, idea in enumerate(state.research_ideas, 1):
        lines.append(f"{i}. **{idea.get('title', 'Untitled')}**")
        desc = idea.get("description", "")
        if desc:
            lines.append(f"   {desc}")
        lines.append("")
    return "\n".join(lines)


def _format_report_summary(state: IdeationAgentState) -> str:
    """Format final ideation report + novelty scores for user review."""
    lines = []
    if state.novelty_score is not None:
        lines.append(f"Average novelty score: {state.novelty_score:.1f}/10\n")
    if state.idea_novelty_assessments:
        for a in state.idea_novelty_assessments:
            title = a.get("title", "Untitled")
            score = a.get("novelty_score", "N/A")
            lines.append(f"  - {title}: {score}/10")
        lines.append("")
    if state.output_summary:
        lines.append(state.output_summary)
    return "\n".join(lines) if lines else "No report generated."


approve_ideas_node, approve_ideas_conditional = make_selection_approval_node(
    node_name="approve_ideas",
    summary_extractor=_format_ideas_summary,
    items_extractor=lambda s: s.research_ideas,
    selection_handler=lambda s, idx: setattr(s, "selected_idea_index", idx),
    retry_target="generate_ideas",
    next_target="novelty_check",
    title="Select a research idea to pursue, or reject to regenerate.",
)

approve_report_node, approve_report_conditional = make_approval_node(
    node_name="approve_report",
    summary_extractor=_format_report_summary,
    retry_target="generate_ideas",
    next_target="end",
    title="Are you satisfied with the ideation report?",
)


@logger.catch
def build():
    """Build ideation agent graph for research ideation through literature review.

    Flow:
    START -> keyword_construct -> literature_search -> analyze_papers -> generate_ideas
      -> approve_ideas -> (generate_ideas[retry] | novelty_check[approved])
      -> novelty_check -> ideation_report -> approve_report -> (generate_ideas[retry] | END[approved])
    """
    g = StateGraph(IdeationAgentState)

    # Nodes
    g.add_node("keyword_construct", execute.keyword_construct_node)
    g.add_node("literature_search", execute.literature_search_node)
    g.add_node("analyze_papers", execute.analyze_papers_node)
    g.add_node("generate_ideas", execute.generate_ideas_node)
    g.add_node("approve_ideas", approve_ideas_node)
    g.add_node("novelty_check", execute.novelty_check_node)
    g.add_node("ideation_report", execute.ideation_report_node)
    g.add_node("approve_report", approve_report_node)

    # Flow
    g.add_edge(START, "keyword_construct")
    g.add_edge("keyword_construct", "literature_search")
    g.add_edge("literature_search", "analyze_papers")
    g.add_edge("analyze_papers", "generate_ideas")

    # Approval after ideas generation
    g.add_edge("generate_ideas", "approve_ideas")
    g.add_conditional_edges(
        "approve_ideas",
        approve_ideas_conditional,
        {
            "generate_ideas": "generate_ideas",
            "novelty_check": "novelty_check",
        },
    )

    g.add_edge("novelty_check", "ideation_report")

    # Approval after final report
    g.add_edge("ideation_report", "approve_report")
    g.add_conditional_edges(
        "approve_report",
        approve_report_conditional,
        {
            "generate_ideas": "generate_ideas",
            "end": END,
        },
    )

    return g
