"""
Build the Experiment Execution Agent graph
"""

from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import execute
from .state import ExecAgentState


def init_node(agent_state: ExecAgentState) -> ExecAgentState:
    """Initialize the agent with the user query as the first message"""
    logger.trace("init_node of ExecAgent")

    # Add the initial user query message if history is empty
    if not agent_state.history or len(agent_state.history) == 0:
        from scievo.core.types import Message
        from scievo.prompts import PROMPTS

        user_msg = Message(
            role="user",
            content=PROMPTS.experiment_exec.exec_user_prompt.render(
                user_query=agent_state.user_query,
                working_dir=agent_state.workspace,
                current_coding_summary=(
                    agent_state.coding_summaries[-1]
                    if agent_state.coding_summaries is not None
                    and len(agent_state.coding_summaries) > 0
                    else None
                ),
                coding_summaries=agent_state.coding_summaries,
            ),
        )
        agent_state.add_message(user_msg)
    else:
        logger.warning("Agent history is not empty during init_node; skipping adding user query.")

    agent_state.intermediate_state.append(
        {
            "node_name": "init",
            "output": user_msg.content if "user_msg" in locals() else "Initialization complete",
        }
    )

    return agent_state


@logger.catch
def build():
    """Build and return the Experiment Execution Agent graph"""
    g = StateGraph(ExecAgentState)

    # Add nodes
    g.add_node("init", init_node)
    g.add_node("gateway", execute.gateway_node)
    g.add_node("llm_chat", execute.llm_chat_node)
    g.add_node("tool_calling", execute.tool_calling_node)
    g.add_node("monitoring", execute.monitoring_node)
    g.add_node("summary", execute.summary_node)
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
            "monitoring",
            "summary",
            "history_compression",
        ],
    )

    # LLM chat -> Gateway
    g.add_edge("llm_chat", "gateway")

    # Tool calling -> Gateway
    g.add_edge("tool_calling", "gateway")

    # Monitoring -> Gateway (after checking/interrupting)
    g.add_edge("monitoring", "gateway")

    # History compression -> Gateway
    g.add_edge("history_compression", "gateway")

    # Summary -> END
    g.add_edge("summary", END)

    return g
