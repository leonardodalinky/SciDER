from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute
from .state import IdeationAgentState


@logger.catch
def build():
    """Build ideation agent graph for research ideation through literature review.

    Flow:
    START -> literature_search -> analyze_papers -> generate_ideas -> ideation_report -> END
    """
    g = StateGraph(IdeationAgentState)

    # Nodes
    g.add_node("literature_search", execute.literature_search_node)
    g.add_node("analyze_papers", execute.analyze_papers_node)
    g.add_node("generate_ideas", execute.generate_ideas_node)
    g.add_node("ideation_report", execute.ideation_report_node)

    # Flow
    g.add_edge(START, "literature_search")
    g.add_edge("literature_search", "analyze_papers")
    g.add_edge("analyze_papers", "generate_ideas")
    g.add_edge("generate_ideas", "ideation_report")
    g.add_edge("ideation_report", END)

    return g
