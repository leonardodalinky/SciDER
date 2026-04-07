"""Build the experiment agent graph.

Flow: START → init → agent_loop → critic_review → user_review → [pass/retry] → generate_summary → END
                                                       ↓ (retry)
                                                   agent_loop

The critic agent independently reviews the experiment agent's work,
then the user can approve, reject, or add additional feedback.
"""

from langgraph.graph import END, START, StateGraph
from loguru import logger

from scider.agents.critic_review import (
    critic_review_node,
    user_review_conditional,
    user_review_node,
)
from scider.core.types import Message

from . import execute
from .state import ExperimentAgentState


def init_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    logger.debug("init_node of ExperimentAgent")

    content = f"Run an experiment based on the following objective:\n\n{agent_state.user_query}"
    content += f"\n\nData summary:\n{agent_state.data_summary}"
    if agent_state.repo_source:
        content += f"\n\nRepository: {agent_state.repo_source}"

    agent_state.add_message(
        Message(role="user", content=content, agent_sender="experiment").with_log()
    )

    agent_state.intermediate_state.append({"node_name": "init", "output": "Starting experiment."})
    return agent_state


@logger.catch
def build():
    """Build the experiment agent graph with critic review + user review loop."""
    g = StateGraph(ExperimentAgentState)

    g.add_node("init", init_node)
    g.add_node("agent_loop", execute.agent_loop_node)
    g.add_node("critic_review", critic_review_node)
    g.add_node("user_review", user_review_node)
    g.add_node("generate_summary", execute.generate_summary_node)

    g.add_edge(START, "init")
    g.add_edge("init", "agent_loop")
    g.add_edge("agent_loop", "critic_review")
    g.add_edge("critic_review", "user_review")
    g.add_conditional_edges("user_review", user_review_conditional)
    g.add_edge("generate_summary", END)

    return g
