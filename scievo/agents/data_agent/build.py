from langgraph.graph import END, START, StateGraph
from loguru import logger

from scievo.core import constant
from scievo.core.types import Message
from scievo.rbank.subgraph import mem_consolidation

from . import execute, plan
from .state import DataAgentState

mem_consolidation_subgraph = mem_consolidation.build()
mem_consolidation_subgraph_compiled = mem_consolidation_subgraph.compile()


def finialize_node(agent_state: DataAgentState) -> DataAgentState:
    """A finalization node to do any final processing before ending the graph."""
    agent_state.intermediate_state.append(
        {
            "node_name": "finalize",
            "output": f"Finalization complete. Plans completed: {len(agent_state.past_plans)}, Remaining: {len(agent_state.remaining_plans)}",
        }
    )
    return agent_state


def prepare_for_talk_mode(agent_state: DataAgentState) -> DataAgentState:
    assert agent_state.talk_mode
    agent_state.remaining_plans = ["Response to users' query."]

    mem_output = "Memory consolidation skipped"
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
            mem_output = "Memory consolidation completed"
        except Exception as e:
            error_msg = f"mem_consolidation_error: {e}"
            agent_state.add_message(
                Message(
                    role="assistant",
                    content=error_msg,
                    agent="noname",
                ).with_log()
            )
            mem_output = error_msg

    agent_state.intermediate_state.append(
        {
            "node_name": "prepare_for_talk_mode",
            "output": mem_output,
        }
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
    # g.add_node("critic", execute.critic_node) # not used for now
    g.add_node("critic_before_replan", execute.critic_node)
    g.add_node("finalize", finialize_node)
    g.add_node("generate_summary", execute.generate_summary_node)
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
            "critic_before_replan",  # plan END
        ],
    )

    # edges from nodes to gateway
    g.add_edge("llm_chat", "gateway")
    g.add_edge("tool_calling", "gateway")
    g.add_edge("mem_extraction", "gateway")
    g.add_edge("history_compression", "gateway")

    g.add_edge("critic_before_replan", "replanner")

    # edges from gateway to replanner
    g.add_conditional_edges(
        "replanner",
        plan.should_replan,
        [
            "gateway",
            "finalize",
        ],
    )
    # edges from nodes to end
    g.add_edge("finalize", "generate_summary")
    g.add_edge("generate_summary", "prepare_for_talk_mode")
    g.add_edge("prepare_for_talk_mode", END)
    return g
