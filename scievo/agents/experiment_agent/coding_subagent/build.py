from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute, plan
from .state import ExperimentAgentState


def prepare_for_talk_mode(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    assert agent_state.talk_mode
    agent_state.remaining_plans = ["Response to users' query."]
    return agent_state


@logger.catch
def build():
    g = StateGraph(ExperimentAgentState)

    # nodes
    g.add_node("planner", plan.planner_node)
    g.add_node("llm_chat", execute.llm_chat_node)
    g.add_node("tool_calling", execute.tool_calling_node)
    g.add_node("replanner", plan.replanner_node)
    g.add_node("gateway", execute.gateway_node)
    g.add_node("report", execute.report_node)

    # edges
    g.add_edge(START, "planner")
    g.add_edge("planner", "gateway")
    g.add_conditional_edges(
        "gateway",
        execute.gateway_conditional,
        [
            "llm_chat",
            "tool_calling",
            "replanner",
            END,
        ],
    )
    g.add_edge("llm_chat", "gateway")
    g.add_edge("tool_calling", "gateway")
    g.add_conditional_edges(
        "replanner",
        plan.should_replan,
        [
            "gateway",
            "report",
        ],
    )
    g.add_edge("report", END)

    return g
