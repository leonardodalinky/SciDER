from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute
from .state import CriticAgentState


@logger.catch
def build():
    g = StateGraph(CriticAgentState)

    # nodes
    g.add_node("create_first_user_msg", execute.create_first_user_msg_node)
    g.add_node("gateway", execute.gateway_node)
    g.add_node("llm_chat", execute.llm_chat_node)
    g.add_node("tool_calling", execute.tool_calling_node)
    g.add_node("summary", execute.summary_node)

    # edges
    g.add_edge(START, "create_first_user_msg")
    g.add_edge("create_first_user_msg", "gateway")
    g.add_conditional_edges(
        "gateway",
        execute.gateway_conditional,
        [
            "llm_chat",
            "tool_calling",
            "summary",
        ],
    )

    # edges from nodes back to gateway
    g.add_edge("llm_chat", "gateway")
    g.add_edge("tool_calling", "gateway")

    # edge from summary to end
    g.add_edge("summary", END)

    return g
