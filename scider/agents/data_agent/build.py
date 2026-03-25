from langgraph.graph import END, START, StateGraph
from loguru import logger

from scider.core import constant
from scider.core.approval import make_approval_node
from scider.core.types import Message
from scider.rbank.subgraph import mem_consolidation

from . import execute
from .paper_subagent import build as paper_subagent_build
from .paper_subagent.state import PaperSearchAgentState
from .state import DataAgentState

mem_consolidation_subgraph = mem_consolidation.build()
mem_consolidation_subgraph_compiled = mem_consolidation_subgraph.compile()

paper_subagent_graph = paper_subagent_build()
paper_subagent_graph_compiled = paper_subagent_graph.compile()


# ==================== Node functions ====================


def init_analysis_node(agent_state: DataAgentState) -> DataAgentState:
    """Inject the initial data analysis prompt into history."""
    agent_state.phase = "analysis"
    agent_state.talk_mode = False
    agent_state.add_message(
        Message(
            role="user",
            content=agent_state.user_query,
            agent_sender="data",
        ).with_log()
    )
    agent_state.intermediate_state.append(
        {"node_name": "init_analysis", "output": "Starting data analysis phase."}
    )
    return agent_state


def init_deep_analysis_node(agent_state: DataAgentState) -> DataAgentState:
    """Inject deep analysis prompt with paper/dataset context."""
    agent_state.phase = "deep_analysis"
    agent_state.talk_mode = False

    paper_context = _format_paper_context(agent_state)
    prompt = (
        "Now deepen your analysis using these related research findings:\n\n"
        f"{paper_context}\n\n"
        "Combine these insights with your earlier data analysis. "
        "Identify connections between the data patterns you found and the research literature. "
        "Provide a more comprehensive and contextualized conclusion."
    )
    agent_state.add_message(
        Message(role="user", content=prompt, agent_sender="data_agent").with_log()
    )
    agent_state.intermediate_state.append(
        {"node_name": "init_deep_analysis", "output": "Starting deep analysis with paper context."}
    )
    return agent_state


def check_phase_node(agent_state: DataAgentState) -> DataAgentState:
    """Mark gateway loop as done for the current phase."""
    agent_state.talk_mode = True
    agent_state.intermediate_state.append(
        {
            "node_name": "check_phase",
            "output": f"Phase '{agent_state.phase}' analysis complete.",
        }
    )
    return agent_state


def check_phase_conditional(agent_state: DataAgentState) -> str:
    if agent_state.phase == "analysis":
        return "approve_analysis"
    else:
        return "approve_final"


def run_paper_subagent(agent_state: DataAgentState) -> DataAgentState:
    """Run paper subagent to search for relevant papers, datasets, and metrics."""
    logger.debug("run_paper_subagent of Agent data")

    # Use analysis output as data_summary for better search relevance
    data_summary = agent_state.data_desc or ""
    for msg in reversed(agent_state.patched_history):
        if msg.role == "assistant" and msg.content:
            data_summary = msg.content[:2000]
            break

    paper_state = PaperSearchAgentState(
        user_query=agent_state.user_query,
        data_summary=data_summary,
    )

    try:
        result_state = paper_subagent_graph_compiled.invoke(paper_state)
        result_state = PaperSearchAgentState(**result_state)

        agent_state.papers = result_state.papers
        agent_state.datasets = result_state.datasets
        agent_state.metrics = result_state.metrics
        agent_state.paper_search_summary = result_state.output_summary

        agent_state.intermediate_state.append(
            {
                "node_name": "paper_subagent",
                "output": (
                    f"Paper subagent completed. Found {len(result_state.papers)} papers, "
                    f"{len(result_state.datasets)} datasets, {len(result_state.metrics)} metrics."
                    f"\n\nSummary: {result_state.output_summary or 'No summary'}"
                ),
            }
        )

        if result_state.output_summary:
            agent_state.add_message(
                Message(
                    role="assistant",
                    content=f"[Paper Search Results]\n{result_state.output_summary}",
                    agent="paper_subagent",
                ).with_log()
            )
    except Exception as e:
        logger.exception("paper_subagent_error")
        error_msg = f"Paper subagent error: {e}"
        agent_state.add_message(
            Message(role="assistant", content=error_msg, agent="paper_subagent").with_log()
        )
        agent_state.intermediate_state.append({"node_name": "paper_subagent", "output": error_msg})

    return agent_state


def finalize_node(agent_state: DataAgentState) -> DataAgentState:
    """Final processing before summary generation."""
    agent_state.intermediate_state.append(
        {"node_name": "finalize", "output": "Finalization complete."}
    )
    return agent_state


def prepare_for_talk_mode(agent_state: DataAgentState) -> DataAgentState:
    assert agent_state.talk_mode

    mem_output = "Memory consolidation skipped"
    if constant.REASONING_BANK_ENABLED:
        try:
            mem_consolidation_subgraph_compiled.invoke(
                mem_consolidation.MemConsolidationState(
                    mem_dir=agent_state.sess_dir / "short_term",
                    long_term_mem_dir=agent_state.long_term_mem_dir,
                    project_mem_dir=agent_state.project_mem_dir,
                )
            )
            mem_output = "Memory consolidation completed"
        except Exception as e:
            error_msg = f"mem_consolidation_error: {e}"
            agent_state.add_message(
                Message(role="assistant", content=error_msg, agent="noname").with_log()
            )
            mem_output = error_msg

    agent_state.intermediate_state.append(
        {"node_name": "prepare_for_talk_mode", "output": mem_output}
    )
    return agent_state


# ==================== Summary extractors for approval ====================


def _format_paper_context(state: DataAgentState) -> str:
    """Format papers/datasets/metrics as context for deep analysis."""
    lines = []
    if state.papers:
        lines.append(f"### Related Papers ({len(state.papers)})")
        for p in state.papers:
            lines.append(f"- **{p.get('title', 'Untitled')}**")
            if p.get("summary"):
                lines.append(f"  {p['summary']}")
    if state.datasets:
        lines.append(f"\n### Related Datasets ({len(state.datasets)})")
        for d in state.datasets:
            lines.append(f"- {d.get('name', 'Unknown')}: {d.get('description', '')}")
    if state.metrics:
        lines.append(f"\n### Evaluation Metrics ({len(state.metrics)})")
        for m in state.metrics:
            lines.append(f"- {m.get('name', 'Unknown')}: {m.get('description', '')}")
    if state.paper_search_summary:
        lines.append(f"\n### Search Summary\n{state.paper_search_summary}")
    return "\n".join(lines) if lines else "No related research found."


def _format_analysis_summary(state: DataAgentState) -> str:
    """Format initial data analysis results for user review."""
    recent = [m for m in state.patched_history[-6:] if m.role == "assistant" and m.content]
    if recent:
        return f"Data analysis result:\n\n{recent[-1].content}"
    return "No analysis output yet."


def _format_paper_results_summary(state: DataAgentState) -> str:
    """Format paper search results for user review."""
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


def _format_final_summary(state: DataAgentState) -> str:
    """Format deep analysis results for final user review."""
    recent = [m for m in state.patched_history[-6:] if m.role == "assistant" and m.content]
    if recent:
        return f"Deep analysis result:\n\n{recent[-1].content}"
    return "No deep analysis output yet."


def _reset_paper_search(state: DataAgentState) -> None:
    """Reset paper search state for retry."""
    state.papers = []
    state.datasets = []
    state.metrics = []
    state.paper_search_summary = None


# ==================== Approval nodes ====================

approve_analysis_node, approve_analysis_conditional = make_approval_node(
    node_name="approve_analysis",
    summary_extractor=_format_analysis_summary,
    retry_target="init_analysis",
    next_target="paper_subagent",
    title="Review the initial data analysis before proceeding to paper search.",
)

approve_papers_node, approve_papers_conditional = make_approval_node(
    node_name="approve_papers",
    summary_extractor=_format_paper_results_summary,
    retry_target="paper_subagent",
    next_target="init_deep_analysis",
    on_retry=_reset_paper_search,
    title="Review the papers, datasets, and metrics found. Proceed to deep analysis?",
)

approve_final_node, approve_final_conditional = make_approval_node(
    node_name="approve_final",
    summary_extractor=_format_final_summary,
    retry_target="init_deep_analysis",
    next_target="finalize",
    title="Are you satisfied with the data analysis output?",
)


# ==================== Gateway conditional (no plan routing) ====================


def gateway_conditional_new(agent_state: DataAgentState) -> str:
    """Route from gateway — same as original but routes assistant → check_phase."""
    result = execute.gateway_conditional(agent_state)
    # gateway_conditional now returns "check_phase" for assistant messages
    return result


# ==================== Build graph ====================


@logger.catch
def build():
    """Build the data agent graph.

    Flow:
    START → init_analysis → gateway loop → check_phase → approve_analysis
      → paper_subagent → approve_papers → init_deep_analysis → gateway loop
      → check_phase → approve_final → finalize → generate_summary
      → prepare_for_talk_mode → END
    """
    g = StateGraph(DataAgentState)

    # Phase 1: Initial data analysis
    g.add_node("init_analysis", init_analysis_node)
    g.add_node("gateway", execute.gateway_node)
    g.add_node("llm_chat", execute.llm_chat_node)
    g.add_node("tool_calling", execute.tool_calling_node)
    g.add_node("mem_extraction", execute.mem_extraction_node)
    g.add_node("history_compression", execute.history_compression_node)
    g.add_node("check_phase", check_phase_node)
    g.add_node("approve_analysis", approve_analysis_node)

    # Paper search
    g.add_node("paper_subagent", run_paper_subagent)
    g.add_node("approve_papers", approve_papers_node)

    # Phase 2: Deep analysis with paper context
    g.add_node("init_deep_analysis", init_deep_analysis_node)
    g.add_node("approve_final", approve_final_node)

    # Finalization
    g.add_node("finalize", finalize_node)
    g.add_node("generate_summary", execute.generate_summary_node)
    g.add_node("prepare_for_talk_mode", prepare_for_talk_mode)

    # --- Edges ---

    # Phase 1 start
    g.add_edge(START, "init_analysis")
    g.add_edge("init_analysis", "gateway")

    # Gateway loop (shared by both phases)
    g.add_conditional_edges(
        "gateway",
        gateway_conditional_new,
        [
            "llm_chat",
            "tool_calling",
            "mem_extraction",
            "history_compression",
            "check_phase",
        ],
    )
    g.add_edge("llm_chat", "gateway")
    g.add_edge("tool_calling", "gateway")
    g.add_edge("mem_extraction", "gateway")
    g.add_edge("history_compression", "gateway")

    # When analysis done → route based on phase
    g.add_conditional_edges(
        "check_phase",
        check_phase_conditional,
        ["approve_analysis", "approve_final"],
    )

    # Phase 1 approval → paper search
    g.add_conditional_edges(
        "approve_analysis",
        approve_analysis_conditional,
        ["init_analysis", "paper_subagent"],
    )

    # Paper search → approval
    g.add_edge("paper_subagent", "approve_papers")
    g.add_conditional_edges(
        "approve_papers",
        approve_papers_conditional,
        ["paper_subagent", "init_deep_analysis"],
    )

    # Phase 2: deep analysis reuses gateway loop
    g.add_edge("init_deep_analysis", "gateway")

    # Phase 2 approval → finalize
    g.add_conditional_edges(
        "approve_final",
        approve_final_conditional,
        ["init_deep_analysis", "finalize"],
    )

    # Finalization chain
    g.add_edge("finalize", "generate_summary")
    g.add_edge("generate_summary", "prepare_for_talk_mode")
    g.add_edge("prepare_for_talk_mode", END)

    return g
