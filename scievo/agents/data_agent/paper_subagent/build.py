from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute
from .state import PaperSearchAgentState


@logger.catch
def build():
    """Build minimal paper search agent graph.

    This is a minimal graph that searches for papers and generates a summary.
    Flow: START -> search_node -> summary_node -> END
    """
    g = StateGraph(PaperSearchAgentState)

    # Nodes - minimal: just search and summary
    g.add_node("search", execute.search_node)
    g.add_node("summary", execute.summary_node)

    # Simple linear flow
    g.add_edge(START, "search")
    g.add_edge("search", "summary")
    g.add_edge("summary", END)

    return g
