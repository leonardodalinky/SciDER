"""Writing agent — runs the PaperOrchestra pipeline in a single agent loop.

Context architecture (identical to data_agent):
- System Prompt: stable identity + workflow (from YAML, cacheable)
- System Context: per-call env (paper_workspace, scider_workspace, date)
- User Context: the top-level task + upstream materials (injected by build.init_node)
"""

from datetime import datetime

from loguru import logger

from scider.core.llms import ModelRegistry
from scider.core.query import QueryResult, gather_tools, query
from scider.core.types import Message
from scider.prompts import PROMPTS

from .state import WritingAgentState

LLM_NAME = "writing"
AGENT_NAME = "writing"

AGENT_TOOLS = [
    "Read",
    "FileEdit",
    "FileWrite",
    "Bash",
    "Glob",
    "Grep",
    "WebSearch",
    "WebFetch",
    # Writing agent does NOT fork sub-agents — it uses the Skill tool to
    # load each PaperOrchestra sub-skill, then runs its instructions inline.
    "AskUserQuestion",
    "TaskOutput",
    "TaskStop",
    "TodoWrite",
    "Skill",
    "RecallHistory",
]


def _get_system_prompt() -> str:
    """Render the writing agent's stable system prompt.

    Only dynamic var is ``supports_vision`` — toggles the visual-verification
    guidance block based on whether the currently-bound model for the
    ``writing`` role accepts image inputs.
    """
    from scider.default.models.catalog import is_vision_model

    return PROMPTS.writing_agent.system_prompt.render(
        supports_vision=is_vision_model(LLM_NAME),
    )


def _build_system_context(agent_state: WritingAgentState) -> str:
    """Build per-session context: paper workspace + SciDER workspace + date."""
    from scider.core.utils import detect_gpu_runtime, detect_python_runtime

    parts = [
        f"paper_workspace (cwd): {agent_state.workspace.working_dir}",
        f"scider_workspace (read-only source): {agent_state.scider_workspace}",
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


def _inject_system_context(agent_state: WritingAgentState) -> None:
    """Inject system context as a meta user message if not already present."""
    if agent_state.history and agent_state.history[0].is_meta:
        return

    context_msg = Message(
        role="user",
        content=_build_system_context(agent_state),
        agent_sender=AGENT_NAME,
        is_meta=True,
    )
    agent_state.history.insert(0, context_msg)


def agent_loop_node(agent_state: WritingAgentState) -> WritingAgentState:
    """Single agent loop — drives the PaperOrchestra 6-step pipeline."""
    logger.debug("agent_loop_node of WritingAgent")
    agent_state.add_node_history("agent_loop")

    _inject_system_context(agent_state)

    system_prompt = _get_system_prompt()
    tools = gather_tools(AGENT_TOOLS)

    # Writing agent needs a much higher autocompact threshold than the default
    # (128k) because section-writing requires holding many files simultaneously
    # (template, idea, experimental_log, outline, refs.bib, intro_relwork, etc.).
    # With 128k the agent enters a read→compact→re-read loop.
    from scider.core.compact import override_compact_settings

    with override_compact_settings(AUTOCOMPACT_TOKEN_THRESHOLD=768_000):
        result: QueryResult = query(
            model_name=LLM_NAME,
            agent_state=agent_state,
            system_prompt=system_prompt,
            tools=tools,
            agent_name=AGENT_NAME,
            tool_execution_context=agent_state.workspace,
            max_turns=200,
        )

    agent_state.intermediate_state.append(
        {
            "node_name": "agent_loop",
            "output": f"Completed in {result.turn_count} turns ({result.stop_reason})",
        }
    )

    # After the loop, try to locate final artifacts on disk so the workflow
    # can surface them even if the agent's own summary forgot to include them.
    _probe_final_artifacts(agent_state)

    return agent_state


def _probe_final_artifacts(agent_state: WritingAgentState) -> None:
    """Best-effort: check paper_workspace/final/ for paper.tex + paper.pdf."""
    ws = agent_state.workspace.working_dir
    final_dir = ws / "final"
    tex = final_dir / "paper.tex"
    pdf = final_dir / "paper.pdf"
    if tex.is_file():
        agent_state.final_tex_path = str(tex.resolve())
    if pdf.is_file():
        agent_state.final_pdf_path = str(pdf.resolve())


def generate_summary_node(agent_state: WritingAgentState) -> WritingAgentState:
    """Ask the model to emit a concise PaperOrchestra execution report."""
    logger.debug("generate_summary_node of WritingAgent")
    agent_state.add_node_history("generate_summary")

    agent_state.add_message(
        Message(
            role="user",
            content=PROMPTS.writing_agent.summary_user_prompt.render(),
            agent_sender=AGENT_NAME,
            is_meta=True,
        )
    )

    msg = ModelRegistry.completion(
        LLM_NAME,
        agent_state.messages,
        system_prompt=PROMPTS.writing_agent.summary_system_prompt.render(),
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
