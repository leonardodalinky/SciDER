"""Data agent — single agent loop, autonomous analysis.

The agent decides what to do: explore data, search papers, present findings.
No rigid phases, no forced approval gates.

Context architecture (following Claude Code's 3-pillar model):
- System Prompt: stable agent identity + guidelines (from YAML, cacheable)
- System Context: per-call environment info (workspace path, injected as message)
- User Context: user query + data description (injected by build.py init_node)
"""

from datetime import datetime

from loguru import logger

# Ensure paper_subagent agent type is registered
import scider.agents.paper_subagent.build  # noqa: F401  # triggers AgentRegistry registration
from scider.core.llms import ModelRegistry
from scider.core.query import QueryResult, gather_tools, query
from scider.core.types import Message
from scider.prompts import PROMPTS

from .state import DataAgentState

LLM_NAME = "data"
AGENT_NAME = "data"

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
    "Skill",
    "RecallHistory",
]


def _get_system_prompt() -> str:
    """Get the stable system prompt (agent identity + guidelines).

    The only dynamic variable is ``supports_vision``, which toggles the
    visual-inspection guidance block based on whether the currently-bound
    model for the ``data`` role accepts image inputs.
    """
    from scider.default.models.catalog import is_vision_model

    return PROMPTS.data_agent.system_prompt.render(
        supports_vision=is_vision_model(LLM_NAME),
    )


def _build_system_context(agent_state: DataAgentState) -> str:
    """Build per-call system context.

    Injected as a <system-reminder> user message at the start of conversation.
    Contains environment info that changes per session.
    """
    from scider.core.utils import detect_gpu_runtime, detect_python_runtime

    parts = [
        f"workspace: {agent_state.workspace.working_dir}",
        f"date: {datetime.now().strftime('%Y-%m-%d')}",
        detect_python_runtime(),
        detect_gpu_runtime(),
    ]
    return (
        "<system-reminder>\n"
        "As you answer, you can use the following context:\n"
        + "\n".join(parts)
        + "\n</system-reminder>"
    )


def _inject_system_context(agent_state: DataAgentState) -> None:
    """Inject system context as a meta user message if not already present."""
    # Only inject once (check if first message is already a system-reminder)
    if agent_state.history and agent_state.history[0].is_meta:
        return

    context_msg = Message(
        role="user",
        content=_build_system_context(agent_state),
        agent_sender=AGENT_NAME,
        is_meta=True,
    )
    # Prepend to history
    agent_state.history.insert(0, context_msg)


def agent_loop_node(agent_state: DataAgentState) -> DataAgentState:
    """The single agent loop — explore, analyze, research, present."""
    logger.debug("agent_loop_node of DataAgent")
    agent_state.add_node_history("agent_loop")

    # Inject per-session context as first message
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


def generate_summary_node(agent_state: DataAgentState) -> DataAgentState:
    """Generate a final structured summary from the analysis."""
    logger.debug("generate_summary_node of DataAgent")
    agent_state.add_node_history("generate_summary")

    agent_state.add_message(
        Message(
            role="user",
            content=PROMPTS.data_agent.summary_user_prompt.render(),
            agent_sender=AGENT_NAME,
            is_meta=True,
        )
    )

    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.messages,
        system_prompt=PROMPTS.data_agent.summary_system_prompt.render(),
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
