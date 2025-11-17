from langgraph.graph import END, START, StateGraph
from loguru import logger

from scievo.agents.experiment_agent.state import ExperimentAgentState

from . import execute, plan


def prepare_for_talk_mode(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    assert agent_state.talk_mode
    agent_state.remaining_plans = ["Response to users' query."]
    return agent_state


@logger.catch
def build():
    g = StateGraph(ExperimentAgentState)

    # nodes
    g.add_node("planner", plan.planner_node)
    g.add_node("tool_calling", execute.tool_calling_node)
    g.add_node("replanner", plan.replanner_node)
    g.add_node("gateway", execute.gateway_node)

    # edges
    g.add_edge(START, "planner")
    g.add_edge("planner", "tool_calling")
    g.add_edge("tool_calling", "gateway")
    g.add_conditional_edges(
        "gateway",
        execute.gateway_conditional,
        [
            "replanner",
            END,
        ],
    )
    g.add_conditional_edges(
        "replanner",
        plan.should_replan,
        [
            "gateway",
            END,
        ],
    )

    return g
