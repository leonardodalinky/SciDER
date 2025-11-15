from langgraph.graph import END, START, StateGraph
from loguru import logger

from scievo.agents.data_agent.state import DataAgentState
from scievo.core import constant
from scievo.core.types import Message
from scievo.rbank.subgraph import mem_consolidation

from . import execute, plan

mem_consolidation_subgraph = mem_consolidation.build()
mem_consolidation_subgraph_compiled = mem_consolidation_subgraph.compile()


def prepare_for_talk_mode(agent_state: DataAgentState) -> DataAgentState:
    assert agent_state.talk_mode
    agent_state.remaining_plans = ["Response to users' query."]

    # consolidate mems
    if constant.REASONING_BANK_ENABLED:
        try:
            mem_consolidation_subgraph_compiled.invoke(
                mem_consolidation.MemConsolidationState(
                    mem_dir=agent_state.sess_dir / "short_term",
                    long_term_mem_dir=agent_state.long_term_mem_dir,
                    project_mem_dir=agent_state.project_mem_dir,
                )
            )
        except Exception as e:
            agent_state.add_message(
                Message(
                    role="assistant",
                    content=f"mem_consolidation_error: {e}",
                    agent="noname",
                ).with_log()
            )
    return agent_state


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
    g.add_node("history_compression", execute.history_compression_node)
    g.add_node("prepare_for_talk_mode", prepare_for_talk_mode)

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
            "history_compression",
            "replanner",  # END
        ],
    )

    # edges from nodes to gateway
    g.add_edge("llm_chat", "gateway")
    g.add_edge("tool_calling", "gateway")
    g.add_edge("mem_extraction", "gateway")
    g.add_edge("history_compression", "gateway")

    # edges from gateway to replanner
    g.add_conditional_edges(
        "replanner",
        plan.should_replan,
        [
            "gateway",
            "prepare_for_talk_mode",
        ],
    )
    g.add_edge("prepare_for_talk_mode", END)
    return g
