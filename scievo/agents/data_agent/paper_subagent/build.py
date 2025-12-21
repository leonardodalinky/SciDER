from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute
from .state import PaperSearchAgentState


@logger.catch
def build():
    """Build minimal paper search agent graph.

    This is a minimal graph that searches for papers, datasets, extracts metrics, and generates a summary.
    Flow: START -> search_node -> dataset_node -> metric_node -> summary_node -> END
    """
    g = StateGraph(PaperSearchAgentState)

    # Nodes - minimal: search papers, search datasets, extract metrics, and summary
    g.add_node("search", execute.search_node)
    g.add_node("dataset", execute.dataset_node)
    g.add_node("metric", execute.metric_node)
    g.add_node("summary", execute.summary_node)

    # Simple linear flow
    g.add_edge(START, "search")
    g.add_edge("search", "dataset")
    g.add_edge("dataset", "metric")
    g.add_edge("metric", "summary")
    g.add_edge("summary", END)

    return g
