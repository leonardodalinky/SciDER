"""Experiment agent — plans, codes, executes, and analyzes experiments.

Single agent loop with tool access for coding, execution, and analysis.
"""

import json

from loguru import logger

# Ensure coding subagent is registered in AgentRegistry
import scider.agents.experiment_agent.coding_subagent  # noqa: F401
from scider.core.agent_utils import inject_system_reminder, today_part
from scider.core.llms import ModelRegistry
from scider.core.query import QueryResult, gather_tools, query
from scider.core.types import Message
from scider.core.utils import wrap_text_with_block
from scider.prompts import PROMPTS

from .state import ExperimentAgentState

LLM_NAME = "experiment"
AGENT_NAME = "experiment"

_BASE_TOOLS = [
    "Read",
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
    "EnterPlanMode",
    "ExitPlanMode",
]

# Tools only available when there's no dedicated coding subagent
_WRITE_TOOLS = ["FileEdit", "FileWrite"]


def _get_coding_backend() -> str:
    import os

    raw = os.getenv("CODING_AGENT_VERSION", "claude_sdk")
    return {"v3": "claude_sdk", "native": "native"}.get(raw, raw)


# Backends that delegate coding to a subagent (experiment agent gets no write tools)
_SUBAGENT_BACKENDS = {"claude_sdk", "native"}


def _get_agent_tools() -> list[str]:
    """Return tool list based on coding backend.

    When a dedicated coding subagent is configured (claude_sdk, native),
    FileEdit/FileWrite are removed — all code writing goes through the subagent.
    """
    backend = _get_coding_backend()
    if backend in _SUBAGENT_BACKENDS:
        return _BASE_TOOLS
    return _BASE_TOOLS + _WRITE_TOOLS


def _get_system_prompt() -> str:
    from scider.default.models.catalog import is_vision_model

    return PROMPTS.experiment_agent.system_prompt.render(
        coding_backend=_get_coding_backend(),
        supports_vision=is_vision_model(LLM_NAME),
    )


def _inject_system_context(agent_state: ExperimentAgentState) -> None:
    from scider.core.utils import detect_gpu_runtime, detect_python_runtime, get_git_status

    parts = [
        f"workspace: {agent_state.workspace.working_dir}",
        today_part(),
        detect_python_runtime(agent_state.workspace.init_config),
        detect_gpu_runtime(),
    ]
    if agent_state.repo_source:
        parts.append(f"repo_source: {agent_state.repo_source}")
    git_status = get_git_status(cwd=str(agent_state.workspace.working_dir))
    if git_status:
        parts.append(f"\n{git_status}")
    inject_system_reminder(
        agent_state.history,
        agent_name=AGENT_NAME,
        parts=parts,
        preamble="As you work, you can use the following context:",
    )


def agent_loop_node(agent_state: ExperimentAgentState) -> ExperimentAgentState:
    logger.debug("agent_loop_node of ExperimentAgent")
    agent_state.add_node_history("agent_loop")

    _inject_system_context(agent_state)

    system_prompt = _get_system_prompt()
    tools = gather_tools(_get_agent_tools())

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


def _is_degenerate_summary(text: str) -> bool:
    """True if a summary call returned a deferral/preamble instead of a report.

    A real report is long and carries markdown section headers. A deferral like
    "Let me first collect the results before writing the report." is short and
    has no headers — catching it lets us retry before it poisons the paper.
    """
    s = (text or "").strip()
    if len(s) < 400:
        return True
    if "#" not in s:  # no markdown headers at all → not the structured report
        return True
    deferral_markers = (
        "let me first",
        "let me collect",
        "i will first",
        "i'll first",
        "before writing the report",
        "before i write",
    )
    head = s[:200].lower()
    return any(m in head for m in deferral_markers)


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

    # The summary call is tool-less and single-shot. Agentic models sometimes
    # reply with a deferral preamble ("Let me first collect the results...")
    # instead of the report itself; that empty summary then propagates into the
    # paper's experimental_log as "(none recorded)". Detect that and retry once
    # with a forceful, no-tools instruction before giving up.
    summary = msg.content or ""
    if _is_degenerate_summary(summary):
        logger.warning(
            "Experiment summary looks degenerate (len={}); retrying once.", len(summary.strip())
        )
        agent_state.add_message(
            Message(
                role="user",
                content=(
                    "That was not a report. Write the COMPLETE experiment report NOW, "
                    "in THIS single message. You have NO tools and cannot run code or "
                    "read files — every result you need is already in the conversation "
                    "above. Do NOT defer, do NOT say you will collect data, do NOT use "
                    "placeholders. Output the full markdown report with all required "
                    "sections and every concrete number from the conversation."
                ),
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
        summary = msg.content or ""

    agent_state.output_summary = summary
    agent_state.final_summary = agent_state.output_summary
    agent_state.final_status = "success"

    agent_state.intermediate_state.append(
        {
            "node_name": "generate_summary",
            "output": agent_state.output_summary[:200] if agent_state.output_summary else "",
        }
    )

    return agent_state
