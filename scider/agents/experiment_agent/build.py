"""
Build the Experiment Agent graph.
"""

from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute
from .state import ExperimentAgentState


@logger.catch
def build():
    """Build the Experiment Agent graph with sub-agent composition."""
    g = StateGraph(ExperimentAgentState)

    # ==================== NODES ====================
    # Initialization node - prepares initial context
    g.add_node("init", execute.init_node)

    # Sub-agent nodes - invoke compiled sub-graphs
    g.add_node("coding", execute.run_coding_subagent)
    g.add_node("exec", execute.run_exec_subagent)
    g.add_node("summary", execute.run_summary_subagent)

    # Analysis node - analyzes loop results and generates insights
    g.add_node("analysis", execute.analysis_node)

    # Revision judge node - decides whether to continue or complete
    g.add_node("revision_judge", execute.revision_judge_node)

    # Finalize node - prepares final output
    g.add_node("finalize", execute.finalize_node)

    # ==================== EDGES ====================
    # Start -> Init
    g.add_edge(START, "init")

    # Init -> Coding
    g.add_edge("init", "coding")

    # Coding -> Exec
    g.add_edge("coding", "exec")

    # Exec -> Summary
    g.add_edge("exec", "summary")

    # Summary -> Analysis
    g.add_edge("summary", "analysis")

    # Analysis -> Revision Judge
    g.add_edge("analysis", "revision_judge")

    # Revision Judge -> Conditional (Continue loop or Complete)
    g.add_conditional_edges(
        "revision_judge",
        execute.should_continue_revision,
        {
            "continue": "coding",  # Go back to coding for next revision
            "complete": "finalize",  # Exit the loop
        },
    )

    # Finalize -> END
    g.add_edge("finalize", END)

    return g
