"""Critic agent — reviews and critiques other agents' work."""

from loguru import logger

from scider.core.agent_utils import build_system_reminder, today_part
from scider.core.llms import ModelRegistry
from scider.core.query import QueryResult, gather_tools, query
from scider.core.types import Message
from scider.prompts import PROMPTS

from .state import CriticAgentState

LLM_NAME = "critic"
AGENT_NAME = "critic"

AGENT_TOOLS = [
    "Read",
    "Bash",
    "Glob",
    "Grep",
    "WebSearch",
    "WebFetch",
    "AskUserQuestion",
    "RecallHistory",
]
# NOTE: Critic has Bash for read-only verification (wc, head, python -c).
# Writing is prohibited by prompt. FileEdit/FileWrite are intentionally excluded.


def _get_system_prompt() -> str:
    return PROMPTS.critic_agent.system_prompt.render()


def _build_system_context(agent_state: CriticAgentState) -> str:
    parts = [today_part()]
    if agent_state.is_data_agent:
        parts.append("reviewing: data agent output")
    if agent_state.is_exp_agent:
        parts.append("reviewing: experiment agent output")
    return build_system_reminder(parts)


def create_first_user_msg_node(agent_state: CriticAgentState) -> CriticAgentState:
    logger.debug("create_first_user_msg_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("create_first_user_msg")

    # Inject system context
    agent_state.add_message(
        Message(
            role="user",
            content=_build_system_context(agent_state),
            agent_sender=AGENT_NAME,
            is_meta=True,
        )
    )

    # Format input messages into the user prompt
    input_msgs_texts = []
    for i, msg in enumerate(agent_state.input_msgs):
        plain = msg.to_plain_text()
        input_msgs_texts.append(f"--- Message {i} Begin ---\n{plain}\n--- Message {i} End ---")
    trajectory_text = "\n".join(input_msgs_texts)

    plan_text = f"\n\nCurrent plan:\n{agent_state.plan}" if agent_state.plan else ""

    agent_state.add_message(
        Message(
            role="user",
            content=f"Please review the following agent trajectory and provide critique.{plan_text}\n\n{trajectory_text}",
            agent_sender=AGENT_NAME,
        )
    )

    return agent_state


def agent_loop_node(agent_state: CriticAgentState) -> CriticAgentState:
    logger.debug("agent_loop_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("agent_loop")

    system_prompt = _get_system_prompt()
    tools = gather_tools(AGENT_TOOLS)

    result: QueryResult = query(
        model_name=LLM_NAME,
        agent_state=agent_state,
        system_prompt=system_prompt,
        tools=tools,
        agent_name=AGENT_NAME,
        tool_execution_context=agent_state.workspace,
    )

    return agent_state


def summary_node(agent_state: CriticAgentState) -> CriticAgentState:
    logger.debug("summary_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("summary")

    agent_state.add_message(
        Message(
            role="user",
            content=PROMPTS.critic_agent.summary_user_prompt.render(),
            agent_sender=AGENT_NAME,
            is_meta=True,
        )
    )

    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.messages,
        system_prompt=PROMPTS.critic_agent.summary_system_prompt.render(),
        agent_sender=AGENT_NAME,
    ).with_log()

    agent_state.add_message(msg)
    agent_state.critic_msg = msg

    return agent_state
