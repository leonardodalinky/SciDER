from langgraph.graph import END, START, StateGraph
from loguru import logger

from scider.tools.agent_tool import AgentRegistry, AgentType

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
            "continue_search": "optimize_query",
            "proceed": "dataset",
        },
    )

    g.add_edge("dataset", "metric")
    g.add_edge("metric", "summary")
    g.add_edge("summary", END)

    return g


# ==================== Tool wrapper ====================

_compiled_graph = build().compile()


def _build_paper_search_state(prompt: str, parent_state) -> dict:
    """Build PaperSearchAgentState kwargs from AgentTool prompt + parent state."""
    data_summary = ""
    if parent_state is not None and hasattr(parent_state, "messages"):
        for msg in reversed(parent_state.messages):
            if msg.role == "assistant" and msg.content:
                data_summary = msg.content[:2000]
                break
    if not data_summary and parent_state is not None and hasattr(parent_state, "data_desc"):
        data_summary = getattr(parent_state, "data_desc", "") or ""

    return {
        "user_query": prompt,
        "data_summary": data_summary,
    }


AgentRegistry.register(
    AgentType(
        name="paper_search",
        description=(
            "Search for academic papers, datasets, and evaluation metrics "
            "related to a research query. Returns papers, datasets, and metrics."
        ),
        compiled_graph=_compiled_graph,
        state_cls=PaperSearchAgentState,
        state_builder=_build_paper_search_state,
        result_extractor=lambda r: {
            "papers": r.get("papers", []),
            "datasets": r.get("datasets", []),
            "metrics": r.get("metrics", []),
            "summary": r.get("output_summary", ""),
        },
        parent_state_updater=lambda ps, res: (
            setattr(ps, "papers", res.get("papers", [])) if hasattr(ps, "papers") else None,
            setattr(ps, "datasets", res.get("datasets", [])) if hasattr(ps, "datasets") else None,
            setattr(ps, "metrics", res.get("metrics", [])) if hasattr(ps, "metrics") else None,
        ),
    )
)
