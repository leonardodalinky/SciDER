from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute
from .state import PaperSearchAgentState


@logger.catch
def build():
    """Build paper search agent graph with iterative query refinement.

    Flow:
    START -> optimize_query -> search -> check_results ->
        (if insufficient results) -> optimize_query -> search -> check_results -> ...
        (if sufficient results) -> dataset -> metric -> summary -> END
    """
    g = StateGraph(PaperSearchAgentState)

    # Nodes
    g.add_node("optimize_query", execute.optimize_query_node)
    g.add_node("search", execute.search_node)
    g.add_node("check_results", execute.check_results_node)
    g.add_node("dataset", execute.dataset_node)
    g.add_node("metric", execute.metric_node)
    g.add_node("summary", execute.summary_node)

    # Flow with iteration support
    g.add_edge(START, "optimize_query")
    g.add_edge("optimize_query", "search")
    g.add_edge("search", "check_results")

    # Conditional edge: continue searching or proceed
    g.add_conditional_edges(
        "check_results",
        execute.should_continue_search,
        {
            "continue_search": "optimize_query",  # Iterate: optimize query and search again
            "proceed": "dataset",  # Proceed with current results
        },
    )

    # Continue with dataset, metric, and summary
    g.add_edge("dataset", "metric")
    g.add_edge("metric", "summary")
    g.add_edge("summary", END)

    return g
