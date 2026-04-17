"""Approval subagent — runs an LLM judge loop and parses a JSON verdict.

Flow: init → agent_loop → parse_verdict → END
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from loguru import logger

from scider.core.query import QueryResult, gather_tools, query
from scider.core.types import Message
from scider.core.utils import parse_json_from_text
from scider.prompts import PROMPTS

from .state import ApprovalSubagentState

LLM_NAME = "approval"
AGENT_NAME = "approval"

AGENT_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "Bash",
    "TodoWrite",
    "TaskOutput",
    "TaskStop",
]


# Per-node task descriptions. Extend here to hook more approval sites.
NODE_APPROVAL_CRITERIA: dict[str, str] = {
    "user_review": (
        "You are a shipping-gate judge for the {parent_agent} agent's output.\n"
        "Your ONLY job: decide whether the produced RESULTS are good enough to "
        "proceed to summary, or whether the agent should retry.\n\n"
        "## Focus on results, NOT process\n"
        "- DO check: do claimed artifacts exist in the workspace? Is the summary "
        "meaningful and conclusive? Did the agent actually address the user's "
        "query?\n"
        "- DO NOT nitpick: transient tool errors that were later resolved, "
        "stylistic issues, suboptimal-but-working approaches.\n\n"
        "## Approve if\n"
        "- Artifacts answer the user's query with reasonable evidence.\n"
        "- Critical claims in the summary are verifiable against workspace files.\n\n"
        "## Reject (with feedback) if\n"
        "- Claimed outputs do not exist, are empty, or are clearly incorrect.\n"
        "- The user's query was NOT addressed (ignored a dimension, wrong "
        "dataset, etc.).\n"
        "- The critic's critical issues are unaddressed AND materially affect "
        "the result."
    ),
}


def _resolve_approval_model() -> str:
    """Return the role name to use. Falls back to 'critic' if 'approval' is not
    registered, so this subagent never fails to find a model."""
    from scider.core.llms import ModelRegistry

    if LLM_NAME in ModelRegistry.instance().models:
        return LLM_NAME
    logger.debug("'approval' model role not registered, falling back to 'critic'")
    return "critic"


def _get_system_prompt() -> str:
    return PROMPTS.approval_subagent.system_prompt.render()


def _build_system_context(agent_state: ApprovalSubagentState) -> str:
    from scider.core.utils import detect_gpu_runtime, detect_python_runtime

    parts = [
        f"date: {datetime.now().strftime('%Y-%m-%d')}",
        f"approval_node: {agent_state.node_name}",
        f"parent_agent: {agent_state.parent_agent or 'unknown'}",
    ]
    if agent_state.workspace_dir is not None:
        parts.append(f"workspace (read-only verification): {agent_state.workspace_dir}")
    parts.append(detect_python_runtime())
    parts.append(detect_gpu_runtime())
    return "<system-reminder>\n" + "\n".join(parts) + "\n</system-reminder>"


def _build_task_prompt(agent_state: ApprovalSubagentState) -> str:
    """Render the approval task: criteria + summary + user query + critic feedback."""
    criteria_tpl = NODE_APPROVAL_CRITERIA.get(
        agent_state.node_name,
        "Judge whether the produced output is good enough to proceed.",
    )
    criteria = criteria_tpl.format(parent_agent=agent_state.parent_agent or "parent")

    sections: list[str] = [
        "# Approval Task",
        criteria,
        "",
        "## User's original query",
        agent_state.user_query or "(not provided)",
        "",
        "## Review summary (includes critic assessment)",
        agent_state.summary or "(empty)",
    ]

    if agent_state.critic_feedback:
        sections += [
            "",
            "## Raw critic feedback",
            agent_state.critic_feedback,
        ]

    sections += [
        "",
        "## Output format (MANDATORY)",
        "After any verification you perform with Read/Glob/Grep/Bash, end your "
        "FINAL assistant message with a JSON block on its own line:",
        "",
        "```json",
        '{"verdict": "approve" | "reject", "feedback": "<if reject, 1-3 '
        'sentences of specific actionable feedback for the agent>"}',
        "```",
        "",
        "Do not use the TaskOutput tool for the verdict — put it in plain text "
        "in your final message. Be decisive: if results are plausibly good, approve.",
    ]
    return "\n".join(sections)


def init_node(agent_state: ApprovalSubagentState) -> ApprovalSubagentState:
    logger.debug("init_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("init")

    # System context (workspace, date, runtime)
    agent_state.add_message(
        Message(
            role="user",
            content=_build_system_context(agent_state),
            agent_sender=AGENT_NAME,
            is_meta=True,
        )
    )

    # The actual approval task prompt
    agent_state.add_message(
        Message(
            role="user",
            content=_build_task_prompt(agent_state),
            agent_sender=AGENT_NAME,
        )
    )
    return agent_state


def agent_loop_node(agent_state: ApprovalSubagentState) -> ApprovalSubagentState:
    logger.debug("agent_loop_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("agent_loop")

    system_prompt = _get_system_prompt()
    tools = gather_tools(AGENT_TOOLS)
    model_role = _resolve_approval_model()

    result: QueryResult = query(
        model_name=model_role,
        agent_state=agent_state,
        system_prompt=system_prompt,
        tools=tools,
        agent_name=AGENT_NAME,
        max_turns=40,
    )

    agent_state.intermediate_state.append(
        {
            "node_name": "agent_loop",
            "output": f"Completed in {result.turn_count} turns ({result.stop_reason})",
        }
    )
    return agent_state


def parse_verdict_node(agent_state: ApprovalSubagentState) -> ApprovalSubagentState:
    """Extract verdict JSON from the final assistant message. Fail-open on error."""
    logger.debug("parse_verdict_node of Agent {}", AGENT_NAME)
    agent_state.add_node_history("parse_verdict")

    last_assistant: Message | None = None
    for msg in reversed(agent_state.messages):
        if msg.role == "assistant" and msg.content:
            last_assistant = msg
            break

    if last_assistant is None:
        logger.warning("parse_verdict: no assistant message found, fail-open approve")
        agent_state.verdict = "approve"
        return agent_state

    try:
        parsed = parse_json_from_text(last_assistant.content or "")
        if not isinstance(parsed, dict):
            raise ValueError(f"expected dict, got {type(parsed).__name__}")
        verdict = parsed.get("verdict", "").strip().lower()
        if verdict not in ("approve", "reject"):
            raise ValueError(f"invalid verdict: {verdict!r}")
        agent_state.verdict = verdict  # type: ignore[assignment]
        agent_state.feedback = parsed.get("feedback") or None
    except Exception as e:
        logger.warning("parse_verdict: failed to parse ({}), fail-open approve", e)
        agent_state.verdict = "approve"
        agent_state.feedback = None

    agent_state.intermediate_state.append(
        {
            "node_name": "parse_verdict",
            "output": f"verdict={agent_state.verdict}",
        }
    )
    return agent_state


def run_approval_subagent(
    *,
    node_name: str,
    summary: str,
    title: str,
    context,  # ApprovalContext — avoid cyclic import
) -> tuple[Literal["approve", "reject"], str | None]:
    """Invoke the approval subagent end-to-end. Returns (verdict, feedback)."""
    from scider.workflows.history_export import capture_messages, save_subagent_history

    from .build import build as _build

    # Lazy-compiled graph (build once per process).
    global _approval_graph  # type: ignore[name-defined]
    try:
        graph = _approval_graph  # type: ignore[name-defined]
    except NameError:
        graph = _build().compile()
        _approval_graph = graph  # type: ignore[assignment]

    workspace_dir: Path | None = context.workspace_dir
    state = ApprovalSubagentState(
        node_name=node_name,
        summary=summary,
        title=title,
        parent_agent=context.parent_agent,
        workspace_dir=workspace_dir,
        critic_feedback=context.critic_feedback,
        user_query=context.user_query,
    )

    with capture_messages() as captured:
        result_dict = graph.invoke(state, {"recursion_limit": 40})

    result_state = ApprovalSubagentState(**result_dict)

    if workspace_dir is not None:
        try:
            save_subagent_history(
                history=list(captured),
                workspace_path=workspace_dir,
                subagent_type="approval",
            )
        except Exception as e:
            logger.warning("Failed to persist approval subagent history: {}", e)

    verdict = result_state.verdict or "approve"
    return verdict, result_state.feedback  # type: ignore[return-value]
