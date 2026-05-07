"""Data agent — single agent loop, autonomous analysis.

The agent decides what to do: explore data, search papers, present findings.
No rigid phases, no forced approval gates.

Context architecture (following Claude Code's 3-pillar model):
- System Prompt: stable agent identity + guidelines (from YAML, cacheable)
- System Context: per-call environment info (workspace path, injected as message)
- User Context: user query + data description (injected by build.py init_node)
"""

from loguru import logger

# Ensure paper_subagent agent type is registered
import scider.agents.paper_subagent.build  # noqa: F401  # triggers AgentRegistry registration
from scider.core.agent_utils import inject_system_reminder, today_part
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


def _inject_system_context(agent_state: DataAgentState) -> None:
    """Inject the env-context system-reminder as a meta user message
    (idempotent, no-op if already present)."""
    from scider.core.utils import detect_gpu_runtime, detect_python_runtime

    parts = [
        f"workspace: {agent_state.workspace.working_dir}",
        today_part(),
        detect_python_runtime(agent_state.workspace.init_config),
        detect_gpu_runtime(),
    ]
    inject_system_reminder(
        agent_state.history,
        agent_name=AGENT_NAME,
        parts=parts,
        preamble="As you answer, you can use the following context:",
    )


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
