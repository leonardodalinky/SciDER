"""Build the native coding subagent graph.

Flow: START → init → agent_loop → generate_summary → END
"""

from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute
from .state import NativeCodingAgentState


@logger.catch
def build():
    """Build the native coding subagent graph."""
    g = StateGraph(NativeCodingAgentState)

    g.add_node("init", execute.init_node)
    g.add_node("agent_loop", execute.agent_loop_node)
    g.add_node("generate_summary", execute.generate_summary_node)

    g.add_edge(START, "init")
    g.add_edge("init", "agent_loop")
    g.add_edge("agent_loop", "generate_summary")
    g.add_edge("generate_summary", END)

    return g
