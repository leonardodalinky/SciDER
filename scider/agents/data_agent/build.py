"""Build the data agent graph.

Flow: START → init → agent_loop → critic_review → [pass/retry] → generate_summary → END
                                       ↓ (retry)
                                   agent_loop

The critic agent independently reviews the data agent's work.
If critical issues are found, the agent retries (up to max_critic_retries times).
"""

from langgraph.graph import END, START, StateGraph
from loguru import logger

from scider.agents.critic_review import critic_review_node, critic_should_retry
from scider.core.types import Message

from . import execute
from .state import DataAgentState


def init_node(agent_state: DataAgentState) -> DataAgentState:
    """Inject the user query and data context into the conversation."""
    logger.debug("init_node of DataAgent")

    content = f"Please analyze the data at the workspace.\n\nUser query: {agent_state.user_query}"
    if agent_state.data_desc:
        content += f"\n\nAdditional context:\n{agent_state.data_desc}"

    agent_state.add_message(Message(role="user", content=content, agent_sender="data").with_log())

    agent_state.intermediate_state.append(
        {"node_name": "init", "output": "Starting data analysis."}
    )
    return agent_state


@logger.catch
def build():
    """Build the data agent graph with critic review loop."""
    g = StateGraph(DataAgentState)

    g.add_node("init", init_node)
    g.add_node("agent_loop", execute.agent_loop_node)
    g.add_node("critic_review", critic_review_node)
    g.add_node("generate_summary", execute.generate_summary_node)

    g.add_edge(START, "init")
    g.add_edge("init", "agent_loop")
    g.add_edge("agent_loop", "critic_review")
    g.add_conditional_edges("critic_review", critic_should_retry)
    g.add_edge("generate_summary", END)

    return g
