"""
Build the Summary Subagent graph
"""

from langgraph.graph import END, START, StateGraph
from loguru import logger

from scievo.core.types import Message
from scievo.prompts import PROMPTS

from . import execute
from .state import SummaryAgentState

AGENT_NAME = "experiment_summary"


def init_node(agent_state: SummaryAgentState) -> SummaryAgentState:
    """Initialize the agent with the user prompt for summary generation"""
    logger.trace("init_node of SummaryAgent")

    # Add the initial user prompt message for summary generation
    # The history should already contain messages from coding/exec subagents
    user_msg = Message(
        role="user",
        content=PROMPTS.experiment_summary.user_prompt.render(
            working_dir=str(agent_state.workspace.working_dir),
            output_path=agent_state.output_path,
        ),
        agent_sender=AGENT_NAME,
    )
    agent_state.add_message(user_msg)

    return agent_state


@logger.catch
def build():
    """Build and return the Summary Subagent graph"""
    g = StateGraph(SummaryAgentState)

    # Add nodes
    g.add_node("init", init_node)
    g.add_node("gateway", execute.gateway_node)
    g.add_node("llm_chat", execute.llm_chat_node)
    g.add_node("tool_calling", execute.tool_calling_node)
    g.add_node("finalize", execute.finalize_node)
    g.add_node("history_compression", execute.history_compression_node)

    # Add edges
    # Start -> Init -> Gateway
    g.add_edge(START, "init")
    g.add_edge("init", "gateway")

    # Gateway -> conditional routing
    g.add_conditional_edges(
        "gateway",
        execute.gateway_conditional,
        [
            "llm_chat",
            "tool_calling",
            "finalize",
            "history_compression",
        ],
    )

    # LLM chat -> Gateway
    g.add_edge("llm_chat", "gateway")

    # Tool calling -> Gateway
    g.add_edge("tool_calling", "gateway")

    # History compression -> Gateway
    g.add_edge("history_compression", "gateway")

    # Finalize -> END
    g.add_edge("finalize", END)

    return g
