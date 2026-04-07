"""Build the critic agent graph.

Flow: START → create_first_user_msg → agent_loop → summary → END
"""

from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute
from .state import CriticAgentState


@logger.catch
def build():
    g = StateGraph(CriticAgentState)

    g.add_node("create_first_user_msg", execute.create_first_user_msg_node)
    g.add_node("agent_loop", execute.agent_loop_node)
    g.add_node("summary", execute.summary_node)

    g.add_edge(START, "create_first_user_msg")
    g.add_edge("create_first_user_msg", "agent_loop")
    g.add_edge("agent_loop", "summary")
    g.add_edge("summary", END)

    return g
