"""Ideation agent — generates novel research ideas from literature.

Single agent loop: searches papers, identifies gaps, generates ideas.
"""

from datetime import datetime

from loguru import logger

# Ensure paper_subagent agent type is registered
import scider.agents.paper_subagent.build  # noqa: F401
from scider.core.llms import ModelRegistry
from scider.core.query import QueryResult, gather_tools, query
from scider.core.types import Message
from scider.prompts import PROMPTS

from .state import IdeationAgentState

LLM_NAME = "ideation"
AGENT_NAME = "ideation"

AGENT_TOOLS = [
    "WebSearch",
    "WebFetch",
    "Agent",
    "EnterPlanMode",
    "ExitPlanMode",
]


def _get_system_prompt() -> str:
    return PROMPTS.ideation_agent.system_prompt.render()


def _build_system_context(agent_state: IdeationAgentState) -> str:
    parts = [f"date: {datetime.now().strftime('%Y-%m-%d')}"]
    if agent_state.research_domain:
        parts.append(f"research_domain: {agent_state.research_domain}")
    return "<system-reminder>\n" + "\n".join(parts) + "\n</system-reminder>"


def _inject_system_context(agent_state: IdeationAgentState) -> None:
    if agent_state.history and agent_state.history[0].is_meta:
        return
    agent_state.history.insert(
        0,
        Message(
            role="user",
            content=_build_system_context(agent_state),
            agent_sender=AGENT_NAME,
            is_meta=True,
        ),
    )


def agent_loop_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    logger.debug("agent_loop_node of IdeationAgent")
    agent_state.add_node_history("agent_loop")

    _inject_system_context(agent_state)

    system_prompt = _get_system_prompt()
    tools = gather_tools(AGENT_TOOLS)

    result: QueryResult = query(
        model_name=LLM_NAME,
        agent_state=agent_state,
        system_prompt=system_prompt,
        tools=tools,
        agent_name=AGENT_NAME,
    )

    agent_state.intermediate_state.append(
        {
            "node_name": "agent_loop",
            "output": f"Completed in {result.turn_count} turns ({result.stop_reason})",
        }
    )

    return agent_state


def generate_report_node(agent_state: IdeationAgentState) -> IdeationAgentState:
    logger.debug("generate_report_node of IdeationAgent")
    agent_state.add_node_history("generate_report")

    agent_state.add_message(
        Message(
            role="user",
            content=PROMPTS.ideation_agent.summary_user_prompt.render(),
            agent_sender=AGENT_NAME,
            is_meta=True,
        )
    )

    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.messages,
        system_prompt=PROMPTS.ideation_agent.summary_system_prompt.render(),
        agent_sender=AGENT_NAME,
    ).with_log()

    agent_state.add_message(msg)
    agent_state.output_summary = msg.content or ""

    agent_state.intermediate_state.append(
        {
            "node_name": "generate_report",
            "output": agent_state.output_summary[:200] if agent_state.output_summary else "",
        }
    )

    return agent_state
