"""
Build the Experiment Agent graph.
"""

from langgraph.graph import END, START, StateGraph
from loguru import logger

from scider.core.approval import ApprovalResult, AutoApprovalHandler
from scider.core.approval import _get_handler as get_approval_handler
from scider.core.approval import make_approval_node
from scider.core.types import Message

from . import execute
from .state import ExperimentAgentState


def _format_coding_summary(state: ExperimentAgentState) -> str:
    """Format latest coding output for user review."""
    if not state.loop_results:
        return "No coding output yet."
    latest = state.loop_results[-1]
    summary = latest.get("coding_summary", "No summary available.")
    lines = [
        f"Revision {state.current_revision + 1}/{state.max_revisions}",
        f"\nCoding summary:\n{summary}",
    ]
    return "\n".join(lines)


def _format_analysis_summary(state: ExperimentAgentState) -> str:
    """Format experiment analysis for user review."""
    lines = [f"Revision {state.current_revision + 1}/{state.max_revisions}\n"]
    if state.revision_analysis:
        lines.append(f"Analysis:\n{state.revision_analysis}")
    if state.revision_summaries:
        lines.append(f"\nLatest summary:\n{state.revision_summaries[-1]}")
    return "\n".join(lines)


approve_code_node, approve_code_conditional = make_approval_node(
    node_name="approve_code",
    summary_extractor=_format_coding_summary,
    retry_target="coding",
    next_target="exec",
    title="Review generated code before execution.",
)


# ==================== Custom experiment approval (3 outcomes) ====================


def approve_experiment_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    """Approval after experiment analysis with 3 outcomes:

    - Continue revising (approved → revision_judge)
    - Finish early (approved + selected finish → finalize)
    - Reject / feedback (retry → coding)

    For AutoApprovalHandler, always continues to revision_judge.
    """
    handler = get_approval_handler()
    summary = _format_analysis_summary(agent_state)

    # Auto-approve: continue revision
    if isinstance(handler, AutoApprovalHandler):
        agent_state.approval_status = "approved"
        return agent_state

    response = handler.request_approval_with_selection(
        node_name="approve_experiment",
        summary=summary,
        items=[
            {"title": "Continue to next revision"},
            {"title": "Finish now — results are good enough"},
        ],
        title="Review experiment results. Continue revising or finish?",
    )

    if response.result == ApprovalResult.APPROVED:
        if response.selected_index == 1:
            # User chose "Finish now"
            agent_state.approval_status = "finish"
        else:
            # User chose "Continue revising" (default)
            agent_state.approval_status = "approved"
    elif response.result == ApprovalResult.REJECTED:
        agent_state.approval_status = "retry"
        agent_state.add_message(
            Message(
                role="user",
                content=(
                    "[User Feedback on approve_experiment]\n"
                    "The user rejected the current output. "
                    "Please try again with a different approach."
                ),
                agent_sender="user_approval",
            ).with_log()
        )
    elif response.result == ApprovalResult.FEEDBACK:
        agent_state.approval_status = "retry"
        agent_state.add_message(
            Message(
                role="user",
                content=(
                    "[User Feedback on approve_experiment]\n"
                    f"{response.feedback}\n\n"
                    "Please revise accordingly."
                ),
                agent_sender="user_approval",
            ).with_log()
        )

    return agent_state


def approve_experiment_conditional(agent_state: ExperimentAgentState) -> str:
    if agent_state.approval_status == "finish":
        return "finalize"
    elif agent_state.approval_status == "approved":
        return "revision_judge"
    else:
        return "coding"


@logger.catch
def build():
    """Build the Experiment Agent graph with sub-agent composition.

    Flow:
    START -> init -> coding -> approve_code -> (coding[retry] | exec[approved])
      -> exec -> summary -> analysis -> approve_experiment -> (coding[retry] | revision_judge[approved])
      -> revision_judge -> (continue->coding | complete->finalize) -> END
    """
    g = StateGraph(ExperimentAgentState)

    # ==================== NODES ====================
    g.add_node("init", execute.init_node)
    g.add_node("coding", execute.run_coding_subagent)
    g.add_node("approve_code", approve_code_node)
    g.add_node("exec", execute.run_exec_subagent)
    g.add_node("summary", execute.run_summary_subagent)
    g.add_node("analysis", execute.analysis_node)
    g.add_node("approve_experiment", approve_experiment_node)
    g.add_node("revision_judge", execute.revision_judge_node)
    g.add_node("finalize", execute.finalize_node)

    # ==================== EDGES ====================
    g.add_edge(START, "init")
    g.add_edge("init", "coding")

    # Approval after coding
    g.add_edge("coding", "approve_code")
    g.add_conditional_edges(
        "approve_code",
        approve_code_conditional,
        {
            "coding": "coding",
            "exec": "exec",
        },
    )

    g.add_edge("exec", "summary")
    g.add_edge("summary", "analysis")

    # Approval after analysis (3 outcomes: continue / finish / retry)
    g.add_edge("analysis", "approve_experiment")
    g.add_conditional_edges(
        "approve_experiment",
        approve_experiment_conditional,
        {
            "coding": "coding",
            "revision_judge": "revision_judge",
            "finalize": "finalize",
        },
    )

    # Revision judge conditional
    g.add_conditional_edges(
        "revision_judge",
        execute.should_continue_revision,
        {
            "continue": "coding",
            "complete": "finalize",
        },
    )

    g.add_edge("finalize", END)

    return g
