from langgraph.graph import END, START, StateGraph
from loguru import logger

from scievo.agents.data_agent.state import DataAgentState

from . import execute, plan


@logger.catch
def build():
    g = StateGraph(DataAgentState)

    # nodes
    g.add_node("planner", plan.planner_node)
    g.add_node("replanner", plan.replanner_node)

    g.add_node("gateway", execute.gateway_node)
    g.add_node("llm_chat", execute.llm_chat_node)
    g.add_node("tool_calling", execute.tool_calling_node)
    g.add_node("mem_extraction", execute.mem_extraction_node)

    # edges from gateway to nodes
    g.add_edge(START, "planner")
    g.add_edge("planner", "gateway")
    g.add_conditional_edges(
        "gateway",
        execute.gateway_conditional,
        [
            "llm_chat",
            "tool_calling",
            "mem_extraction",
            "replanner",  # END
        ],
    )

    # edges from nodes to gateway
    g.add_edge("llm_chat", "gateway")
    g.add_edge("tool_calling", "gateway")
    g.add_edge("mem_extraction", "gateway")

    # edges from gateway to replanner
    g.add_conditional_edges(
        "replanner",
        plan.should_replan,
        [
            "gateway",
            END,
        ],
    )
    return g
