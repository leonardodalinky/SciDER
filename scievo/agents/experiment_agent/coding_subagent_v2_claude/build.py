from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute, plan
from .state import CodingAgentState


def prepare_for_completion(agent_state: CodingAgentState) -> CodingAgentState:
    """Prepare for final completion after all plans are done."""
    assert agent_state.talk_mode

    # Generate a summary from the last assistant message or history
    # This provides a summary for the experiment agent to use
    if agent_state.patched_history:
        # Try to extract summary from the last assistant message
        last_assistant_msg = None
        for msg in reversed(agent_state.patched_history):
            if msg.role == "assistant" and msg.content:
                last_assistant_msg = msg
                break

        if last_assistant_msg:
            # Use the last assistant message content as summary
            agent_state.output_summary = last_assistant_msg.content[:500]  # Limit to 500 chars
            logger.info(
                f"Coding task summary generated: {len(agent_state.output_summary)} characters"
            )
        else:
            # Fallback: create a simple summary from history
            agent_state.output_summary = (
                f"Coding task completed. Total messages: {len(agent_state.patched_history)}"
            )
    else:
        agent_state.output_summary = "Coding task completed (no history available)"

    return agent_state


@logger.catch
def build():
    g = StateGraph(CodingAgentState)

    # Nodes
    g.add_node("planner", plan.planner_node)
    g.add_node("replanner", plan.replanner_node)

    g.add_node("gateway", execute.gateway_node)
    g.add_node("llm_chat", execute.llm_chat_node)
    g.add_node("tool_calling", execute.tool_calling_node)

    # Critic node - evaluates each plan step
    g.add_node("critic_before_replan", execute.critic_node)

    g.add_node("prepare_for_completion", prepare_for_completion)

    # Edges
    g.add_edge(START, "planner")
    g.add_edge("planner", "gateway")

    g.add_conditional_edges(
        "gateway",
        execute.gateway_conditional,
        [
            "llm_chat",
            "tool_calling",
            "critic_before_replan",  # Plan step END
        ],
    )

    # Edges from nodes back to gateway
    g.add_edge("llm_chat", "gateway")
    g.add_edge("tool_calling", "gateway")

    # Critic leads to replanner
    g.add_edge("critic_before_replan", "replanner")

    # Replanner conditional edges
    g.add_conditional_edges(
        "replanner",
        plan.should_replan,
        [
            "gateway",
            "prepare_for_completion",
        ],
    )

    g.add_edge("prepare_for_completion", END)

    return g
