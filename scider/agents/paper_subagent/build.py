from langgraph.graph import END, START, StateGraph
from loguru import logger

from scider.core.approval import make_approval_node
from scider.tools.agent_tool import AgentRegistry, AgentType

from . import execute
from .state import PaperSearchAgentState


def _format_paper_results_summary(state: PaperSearchAgentState) -> str:
    """Format papers/datasets/metrics summary for user review."""
    lines = [f"Papers found: {len(state.papers)}"]
    for p in state.papers:
        lines.append(f"  - {p.get('title', 'Unknown')}")
    lines.append(f"\nDatasets found: {len(state.datasets)}")
    for d in state.datasets:
        lines.append(f"  - {d.get('name', 'Unknown')}")
    lines.append(f"\nMetrics extracted: {len(state.metrics)}")
    for m in state.metrics:
        lines.append(f"  - {m.get('name', 'Unknown')}")
    return "\n".join(lines)


def _reset_paper_search(state: PaperSearchAgentState) -> None:
    """Reset search state for retry."""
    state.search_iteration = 0
    state.papers = []
    state.datasets = []
    state.metrics = []


approve_results_node, approve_results_conditional = make_approval_node(
    node_name="approve_results",
    summary_extractor=_format_paper_results_summary,
    retry_target="optimize_query",
    next_target="summary",
    on_retry=_reset_paper_search,
    title="Review paper search results. Approve to continue or reject to refine the search.",
)


@logger.catch
def build():
    """Build paper search agent graph with iterative query refinement.

    Flow:
    START -> optimize_query -> search -> check_results ->
        (if insufficient results) -> optimize_query -> search -> check_results -> ...
        (if sufficient results) -> dataset -> metric -> approve_results ->
            (if approved) -> summary -> END
            (if rejected) -> optimize_query (restart search)
    """
    g = StateGraph(PaperSearchAgentState)

    # Nodes
    g.add_node("optimize_query", execute.optimize_query_node)
    g.add_node("search", execute.search_node)
    g.add_node("check_results", execute.check_results_node)
    g.add_node("dataset", execute.dataset_node)
    g.add_node("metric", execute.metric_node)
    g.add_node("approve_results", approve_results_node)
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

    # Continue with dataset, metric, approval, and summary
    g.add_edge("dataset", "metric")
    g.add_edge("metric", "approve_results")
    g.add_conditional_edges(
        "approve_results",
        approve_results_conditional,
        {
            "optimize_query": "optimize_query",  # Retry: restart search
            "summary": "summary",  # Approved: proceed to summary
        },
    )
    g.add_edge("summary", END)

    return g


# ==================== Tool wrapper ====================

_compiled_graph = build().compile()


def _build_paper_search_state(prompt: str, parent_state) -> dict:
    """Build PaperSearchAgentState kwargs from AgentTool prompt + parent state."""
    # Extract data_summary from parent's conversation
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
