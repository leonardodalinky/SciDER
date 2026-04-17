"""Native coding subagent — uses SciDER's query() loop with full tool access.

No external dependencies (no Claude Agent SDK).
Works with any LiteLLM-supported model via the experiment_coding role.
"""

from datetime import datetime

from loguru import logger

from scider.core.llms import ModelRegistry
from scider.core.query import QueryResult, gather_tools, query
from scider.core.types import Message
from scider.prompts import PROMPTS

from .state import NativeCodingAgentState

LLM_NAME = "experiment_coding"
AGENT_NAME = "native_coding"

AGENT_TOOLS = [
    "Read",
    "FileEdit",
    "FileWrite",
    "Bash",
    "Glob",
    "Grep",
    "TodoWrite",
    "WebSearch",
    "WebFetch",
    "AskUserQuestion",
    "Skill",
]


def _get_system_prompt() -> str:
    from scider.default.models.catalog import is_vision_model

    return PROMPTS.coding_subagent_native.system_prompt.render(
        supports_vision=is_vision_model(LLM_NAME),
    )


def _build_system_context(agent_state: NativeCodingAgentState) -> str:
    from scider.core.utils import detect_gpu_runtime, detect_python_runtime

    parts = [
        f"workspace: {agent_state.workspace.working_dir}",
        f"date: {datetime.now().strftime('%Y-%m-%d')}",
        detect_python_runtime(),
        detect_gpu_runtime(),
    ]
    return (
        "<system-reminder>\n"
        "As you work, you can use the following context:\n"
        + "\n".join(parts)
        + "\n</system-reminder>"
    )


def _inject_system_context(agent_state: NativeCodingAgentState) -> None:
    """Inject system context as a meta user message if not already present."""
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


def init_node(agent_state: NativeCodingAgentState) -> NativeCodingAgentState:
    """Inject the coding task and background info into the conversation."""
    logger.debug("init_node of NativeCodingAgent")

    content = "Please complete the following coding task in the workspace."
    if agent_state.user_query:
        content += f"\n\n## Task\n{agent_state.user_query}"
    if agent_state.data_summary:
        content += f"\n\n## Background Information\n{agent_state.data_summary}"

    agent_state.add_message(
        Message(role="user", content=content, agent_sender=AGENT_NAME).with_log()
    )

    agent_state.intermediate_state.append({"node_name": "init", "output": "Starting coding task."})
    return agent_state


def agent_loop_node(agent_state: NativeCodingAgentState) -> NativeCodingAgentState:
    """The main coding loop — read, write, test, iterate."""
    logger.debug("agent_loop_node of NativeCodingAgent")
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


def generate_summary_node(
    agent_state: NativeCodingAgentState,
) -> NativeCodingAgentState:
    """Generate a structured summary of the coding work."""
    logger.debug("generate_summary_node of NativeCodingAgent")
    agent_state.add_node_history("generate_summary")

    agent_state.add_message(
        Message(
            role="user",
            content=PROMPTS.coding_subagent_native.summary_user_prompt.render(),
            agent_sender=AGENT_NAME,
            is_meta=True,
        )
    )

    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.messages,
        system_prompt=PROMPTS.coding_subagent_native.summary_system_prompt.render(),
        agent_sender=AGENT_NAME,
    ).with_log()

    agent_state.add_message(msg)
    agent_state.output_summary = msg.content or ""

    agent_state.intermediate_state.append(
        {
            "node_name": "generate_summary",
            "output": agent_state.output_summary[:200] if agent_state.output_summary else "",
        }
    )

    return agent_state
