"""Build the approval subagent graph.

Flow: START → init → agent_loop → parse_verdict → END
"""

from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute
from .state import ApprovalSubagentState


@logger.catch
def build():
    g = StateGraph(ApprovalSubagentState)

    g.add_node("init", execute.init_node)
    g.add_node("agent_loop", execute.agent_loop_node)
    g.add_node("parse_verdict", execute.parse_verdict_node)

    g.add_edge(START, "init")
    g.add_edge("init", "agent_loop")
    g.add_edge("agent_loop", "parse_verdict")
    g.add_edge("parse_verdict", END)

    return g
