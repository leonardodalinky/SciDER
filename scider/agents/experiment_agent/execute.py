"""Experiment agent — plans, codes, executes, and analyzes experiments.

Single agent loop with tool access for coding, execution, and analysis.
"""

import json
from datetime import datetime

from loguru import logger

# Ensure coding subagent is registered in AgentRegistry
import scider.agents.experiment_agent.coding_subagent  # noqa: F401
from scider.core.llms import ModelRegistry
from scider.core.query import QueryResult, gather_tools, query
from scider.core.types import Message
from scider.core.utils import wrap_text_with_block
from scider.prompts import PROMPTS

from .state import ExperimentAgentState

LLM_NAME = "experiment"
AGENT_NAME = "experiment"

AGENT_TOOLS = [
    "Read",
    "FileEdit",
    "FileWrite",
    "Bash",
    "Glob",
    "Grep",
    "WebSearch",
    "WebFetch",
    "Agent",
    "AskUserQuestion",
    "TaskOutput",
    "TaskStop",
    "TodoWrite",
    "RecallHistory",
    "EnterPlanMode",
    "ExitPlanMode",
]


def _get_system_prompt() -> str:
    return PROMPTS.experiment_agent.system_prompt.render()


def _build_system_context(agent_state: ExperimentAgentState) -> str:
    parts = [
        f"workspace: {agent_state.workspace.working_dir}",
        f"date: {datetime.now().strftime('%Y-%m-%d')}",
    ]
    if agent_state.repo_source:
        parts.append(f"repo_source: {agent_state.repo_source}")

    # Git status snapshot (if workspace is a git repo)
    from scider.core.utils import get_git_status

    git_status = get_git_status(cwd=str(agent_state.workspace.working_dir))
    if git_status:
        parts.append(f"\n{git_status}")

    return (
        "<system-reminder>\n"
        "As you work, you can use the following context:\n"
        + "\n".join(parts)
        + "\n</system-reminder>"
    )


def _inject_system_context(agent_state: ExperimentAgentState) -> None:
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


def agent_loop_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    logger.debug("agent_loop_node of ExperimentAgent")
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
        tool_execution_context=agent_state.workspace,
    )

    agent_state.intermediate_state.append(
        {
            "node_name": "agent_loop",
            "output": f"Completed in {result.turn_count} turns ({result.stop_reason})",
        }
    )

    return agent_state


def generate_summary_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    logger.debug("generate_summary_node of ExperimentAgent")
    agent_state.add_node_history("generate_summary")

    agent_state.add_message(
        Message(
            role="user",
            content=PROMPTS.experiment_agent.summary_user_prompt.render(),
            agent_sender=AGENT_NAME,
            is_meta=True,
        )
    )

    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.messages,
        system_prompt=PROMPTS.experiment_agent.summary_system_prompt.render(),
        agent_sender=AGENT_NAME,
    ).with_log()

    agent_state.add_message(msg)
    agent_state.output_summary = msg.content or ""
    agent_state.final_summary = agent_state.output_summary
    agent_state.final_status = "success"

    agent_state.intermediate_state.append(
        {
            "node_name": "generate_summary",
            "output": agent_state.output_summary[:200] if agent_state.output_summary else "",
        }
    )

    return agent_state
