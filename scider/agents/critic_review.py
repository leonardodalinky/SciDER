"""Shared critic review logic for data and experiment agents.

Invokes the critic agent as a compiled subgraph, extracts feedback,
and injects it into the parent agent's conversation for retry.

Used as a graph node in both data_agent and experiment_agent:
    agent_loop → critic_review → [pass/retry] → generate_summary / agent_loop
"""

from __future__ import annotations

from loguru import logger

from scider.agents.critic_agent import build as critic_build
from scider.agents.critic_agent.state import CriticAgentState
from scider.core.types import Message

# Compile critic graph once at import time
_critic_graph = critic_build().compile()


def critic_review_node(agent_state):
    """Run critic agent to review the current agent's work.

    Reads the agent's conversation history, invokes the critic,
    and stores feedback in agent_state.critic_feedback.

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


def _extract_verdict(feedback: str) -> str:
    """Extract verdict from critic feedback.

    Returns 'pass' if the work is acceptable, 'retry' if it needs improvement.
    """
    feedback_lower = feedback.lower()
    # Look for explicit verdict markers
    if "overall assessment" in feedback_lower:
        # Find the line with the assessment
        for line in feedback.splitlines():
            if "overall assessment" in line.lower():
                line_lower = line.lower()
                if "strong" in line_lower or "adequate" in line_lower or "good" in line_lower:
                    return "pass"
                if "poor" in line_lower or "needs improvement" in line_lower:
                    return "retry"

    # Fallback: check for critical issues
    if "critical" in feedback_lower and ("issue" in feedback_lower or "error" in feedback_lower):
        return "retry"

    # Default: pass (don't block on ambiguous feedback)
    return "pass"


def critic_should_retry(agent_state) -> str:
    """Conditional edge: decide whether to retry or proceed to summary.

    Returns 'agent_loop' to retry, or 'generate_summary' to proceed.
    """
    feedback = getattr(agent_state, "critic_feedback", None)
    retry_count = getattr(agent_state, "critic_retry_count", 0)
    max_retries = getattr(agent_state, "max_critic_retries", 2)

    # No feedback or critic failed → proceed
    if not feedback:
        return "generate_summary"

    # Max retries reached → proceed
    if retry_count >= max_retries:
        logger.info("Max critic retries ({}) reached, proceeding to summary", max_retries)
        return "generate_summary"

    # Check verdict
    verdict = _extract_verdict(feedback)
    if verdict == "pass":
        return "generate_summary"

    # Retry: inject feedback into conversation and increment counter
    agent_state.critic_retry_count = retry_count + 1
    agent_state.add_message(
        Message(
            role="user",
            content=(
                f"<system-reminder>\n"
                f"[Critic Review — Retry {agent_state.critic_retry_count}/{max_retries}]\n"
                f"The following issues were found in your work. Please address them:\n\n"
                f"{feedback}\n"
                f"</system-reminder>"
            ),
            agent_sender="critic",
            is_meta=True,
        )
    )
    logger.info("Critic requested retry ({}/{})", agent_state.critic_retry_count, max_retries)
    return "agent_loop"
