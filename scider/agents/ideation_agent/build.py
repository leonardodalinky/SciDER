from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute
from .state import IdeationAgentState


@logger.catch
def build():
    """Build ideation agent graph for research ideation through literature review.

    Flow:
    START -> keyword_construct -> literature_search -> analyze_papers -> generate_ideas -> novelty_check -> ideation_report -> END
    """
    g = StateGraph(IdeationAgentState)

    # Nodes
    g.add_node("keyword_construct", execute.keyword_construct_node)
    g.add_node("literature_search", execute.literature_search_node)
    g.add_node("analyze_papers", execute.analyze_papers_node)
    g.add_node("generate_ideas", execute.generate_ideas_node)
    g.add_node("novelty_check", execute.novelty_check_node)
    g.add_node("ideation_report", execute.ideation_report_node)

    # Flow
    g.add_edge(START, "keyword_construct")
    g.add_edge("keyword_construct", "literature_search")
    g.add_edge("literature_search", "analyze_papers")
    g.add_edge("analyze_papers", "generate_ideas")
    g.add_edge("generate_ideas", "novelty_check")
    g.add_edge("novelty_check", "ideation_report")
    g.add_edge("ideation_report", END)

    return g
