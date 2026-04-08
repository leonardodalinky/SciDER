"""Shared critic review logic for data and experiment agents.

Invokes the critic agent as a compiled subgraph, extracts feedback,
then presents the result to the user for optional additional input.

Used as graph nodes in both data_agent and experiment_agent:
    agent_loop → critic_review → user_review → [pass/retry] → generate_summary / agent_loop
"""

from __future__ import annotations

from loguru import logger

from scider.agents.critic_agent import build as critic_build
from scider.agents.critic_agent.state import CriticAgentState
from scider.core.approval import ApprovalResult, _get_handler
from scider.core.types import Message

# Compile critic graph once at import time
_critic_graph = critic_build().compile()


def critic_review_node(agent_state):
    """Run critic agent to review the current agent's work.

    Accepts any state type (DataAgentState, ExperimentAgentState, etc.)
    that has critic_feedback and intermediate_state fields.
    """
    agent_name = "data" if hasattr(agent_state, "data_desc") else "experiment"
    logger.info("Running critic review for {} agent", agent_name)

    if hasattr(agent_state, "add_node_history"):
        agent_state.add_node_history("critic_review")

    # Build critic state from parent's messages
    critic_state = CriticAgentState(
        input_msgs=agent_state.messages[-20:],  # last 20 messages for context
        is_data_agent=(agent_name == "data"),
        is_exp_agent=(agent_name == "experiment"),
    )

    try:
        result = _critic_graph.invoke(critic_state, {"recursion_limit": 30})
        result_state = CriticAgentState(**result)

        # Extract critic feedback
        feedback = ""
        if result_state.critic_msg and result_state.critic_msg.content:
            feedback = result_state.critic_msg.content

        agent_state.critic_feedback = feedback

        # Persist critic's full conversation history under <workspace>/subagents/
        try:
            workspace = getattr(agent_state, "workspace", None)
            if workspace is not None and hasattr(workspace, "working_dir"):
                from scider.workflows.history_export import save_subagent_history

                save_subagent_history(
                    history=result_state.history,
                    workspace_path=workspace.working_dir,
                    subagent_type="critic",
                )
        except Exception as e:
            logger.warning("Failed to persist critic history: {}", e)

        # Determine verdict
        verdict = _extract_verdict(feedback)
        logger.info("Critic verdict for {} agent: {}", agent_name, verdict)

        if hasattr(agent_state, "intermediate_state"):
            agent_state.intermediate_state.append(
                {
                    "node_name": "critic_review",
                    "output": f"Verdict: {verdict}\n{feedback[:300]}",
                }
            )

    except Exception as e:
        logger.warning("Critic review failed: {}. Skipping.", e)
        agent_state.critic_feedback = None
        if hasattr(agent_state, "intermediate_state"):
            agent_state.intermediate_state.append(
                {
                    "node_name": "critic_review",
                    "output": f"Critic review failed: {e}. Proceeding without review.",
                }
            )

    return agent_state


def user_review_node(agent_state):
    """Present critic feedback to the user and ask for additional input.

    Uses the approval handler to show the critic's assessment and give
    the user a chance to approve, reject, or add their own feedback.
    """
    if hasattr(agent_state, "add_node_history"):
        agent_state.add_node_history("user_review")

    feedback = getattr(agent_state, "critic_feedback", None)
    retry_count = getattr(agent_state, "critic_retry_count", 0)
    max_retries = getattr(agent_state, "max_critic_retries", 2)

    # No feedback or max retries reached → auto-approve
    if not feedback or retry_count >= max_retries:
        agent_state.approval_status = "approved"
        return agent_state

    # Build summary for the user
    verdict = _extract_verdict(feedback)
    summary = f"## Critic Assessment: {verdict.upper()}\n\n{feedback}"
    if verdict == "retry":
        summary += (
            "\n\n---\n"
            "*The critic found issues. You can:*\n"
            "- **Approve** to proceed anyway\n"
            "- **Reject** to retry with the critic's feedback\n"
            "- **Feedback** to add your own instructions for the retry"
        )
    else:
        summary += (
            "\n\n---\n"
            "*The critic thinks this is acceptable. You can:*\n"
            "- **Approve** to proceed to summary\n"
            "- **Feedback** to request additional changes"
        )

    handler = _get_handler()
    response = handler.request_approval(
        node_name="user_review",
        summary=summary,
        title="Review critic feedback — any additional changes?",
    )

    if response.result == ApprovalResult.APPROVED:
        agent_state.approval_status = "approved"
    elif response.result == ApprovalResult.REJECTED:
        agent_state.approval_status = "retry"
        agent_state.critic_retry_count = retry_count + 1
        agent_state.add_message(
            Message(
                role="user",
                content=(
                    f"<system-reminder>\n"
                    f"[Critic Review — Retry {agent_state.critic_retry_count}/{max_retries}]\n"
                    f"The following issues were found. You MUST address ALL of them "
                    f"thoroughly using multiple tool calls before finishing. "
                    f"Do NOT stop after a single tool call — work through each issue.\n\n"
                    f"{feedback}\n"
                    f"</system-reminder>"
                ),
                agent_sender="critic",
                is_meta=True,
            )
        )
    elif response.result == ApprovalResult.FEEDBACK:
        agent_state.approval_status = "retry"
        agent_state.critic_retry_count = retry_count + 1
        user_feedback = response.feedback or ""
        agent_state.add_message(
            Message(
                role="user",
                content=(
                    f"<system-reminder>\n"
                    f"[Review — Retry {agent_state.critic_retry_count}/{max_retries}]\n"
                    f"Address ALL of the following feedback thoroughly using multiple "
                    f"tool calls before finishing.\n\n"
                    f"Critic feedback:\n{feedback}\n\n"
                    f"User additional feedback:\n{user_feedback}\n"
                    f"</system-reminder>"
                ),
                agent_sender="critic",
                is_meta=True,
            )
        )

    if hasattr(agent_state, "intermediate_state"):
        agent_state.intermediate_state.append(
            {
                "node_name": "user_review",
                "output": f"User decision: {agent_state.approval_status}",
            }
        )

    return agent_state


def user_review_conditional(agent_state) -> str:
    """Conditional edge after user_review: retry or proceed."""
    status = getattr(agent_state, "approval_status", "approved")
    if status == "retry":
        return "agent_loop"
    return "generate_summary"


def _extract_verdict(feedback: str) -> str:
    """Extract verdict from critic feedback.

    Returns 'pass' if the work is acceptable, 'retry' if it needs improvement.
    """
    feedback_lower = feedback.lower()
    # Look for explicit verdict markers near "overall assessment"
    if "overall assessment" in feedback_lower:
        lines = feedback.splitlines()
        for i, line in enumerate(lines):
            if "overall assessment" in line.lower():
                # Check this line and the next 2 lines for verdict keywords
                window = " ".join(lines[i : i + 3]).lower()
                if "strong" in window or "adequate" in window or "good" in window:
                    return "pass"
                if "poor" in window or "needs improvement" in window:
                    return "retry"

    # Fallback: check for critical issues
    if "critical" in feedback_lower and ("issue" in feedback_lower or "error" in feedback_lower):
        return "retry"

    # Default: pass (don't block on ambiguous feedback)
    return "pass"
