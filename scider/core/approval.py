"""User approval mechanism for agent workflows.

Provides interactive checkpoints where users can approve, reject, or provide
feedback at critical agent steps. Controlled by USER_APPROVAL_ENABLED env var.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable

from loguru import logger

from scider.core import constant
from scider.core.types import Message


class ApprovalResult(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    FEEDBACK = "feedback"


class ApprovalResponse:
    """Response from user approval request."""

    def __init__(self, result: ApprovalResult, feedback: str | None = None):
        self.result = result
        self.feedback = feedback


class ApprovalHandler(ABC):
    """Abstract base class for approval handlers."""

    @abstractmethod
    def request_approval(self, node_name: str, summary: str) -> ApprovalResponse:
        """Request user approval. Blocks until user responds.

        Args:
            node_name: Name of the node that produced the output.
            summary: Human-readable summary of the output to review.

        Returns:
            ApprovalResponse with the user's decision and optional feedback.
        """
        ...


class CLIApprovalHandler(ApprovalHandler):
    """CLI-based approval handler using input() for blocking user interaction."""

    def request_approval(self, node_name: str, summary: str) -> ApprovalResponse:
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"  User Approval Required: [{node_name}]")
        print(separator)
        print(summary)
        print(separator)
        print("  [1] Approve and continue (default)")
        print("  [2] Reject and retry")
        print("  [3] Provide feedback and retry")
        print(separator)

        while True:
            choice = input("Your choice [1/2/3] (enter = approve): ").strip()
            if choice in ("1", ""):
                logger.info("User approved [{}]", node_name)
                return ApprovalResponse(result=ApprovalResult.APPROVED)
            elif choice == "2":
                logger.info("User rejected [{}]", node_name)
                return ApprovalResponse(result=ApprovalResult.REJECTED)
            elif choice == "3":
                feedback = input("Your feedback (enter = cancel): ").strip()
                if not feedback:
                    print("Feedback cancelled, returning to choices.")
                    continue
                logger.info("User provided feedback for [{}]: {}", node_name, feedback)
                return ApprovalResponse(result=ApprovalResult.FEEDBACK, feedback=feedback)
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")


class JupyterApprovalHandler(ApprovalHandler):
    """Jupyter notebook approval handler with rich display."""

    def request_approval(self, node_name: str, summary: str) -> ApprovalResponse:
        from IPython.display import Markdown, display

        display(Markdown(f"---\n### User Approval Required: `{node_name}`\n\n{summary}\n\n---"))
        print("[1] Approve and continue (default)")
        print("[2] Reject and retry")
        print("[3] Provide feedback and retry")

        while True:
            choice = input("Your choice [1/2/3] (enter = approve): ").strip()
            if choice in ("1", ""):
                logger.info("User approved [{}]", node_name)
                return ApprovalResponse(result=ApprovalResult.APPROVED)
            elif choice == "2":
                logger.info("User rejected [{}]", node_name)
                return ApprovalResponse(result=ApprovalResult.REJECTED)
            elif choice == "3":
                feedback = input("Your feedback (enter = cancel): ").strip()
                if not feedback:
                    print("Feedback cancelled, returning to choices.")
                    continue
                logger.info("User provided feedback for [{}]: {}", node_name, feedback)
                return ApprovalResponse(result=ApprovalResult.FEEDBACK, feedback=feedback)
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")


class AutoApprovalHandler(ApprovalHandler):
    """Auto-approval handler. Always approves without user interaction."""

    def request_approval(self, node_name: str, summary: str) -> ApprovalResponse:
        logger.debug("Auto-approved [{}]", node_name)
        return ApprovalResponse(result=ApprovalResult.APPROVED)


def _is_jupyter() -> bool:
    """Detect if running inside a Jupyter notebook / IPython kernel."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def get_default_handler() -> ApprovalHandler:
    """Get the default approval handler based on USER_APPROVAL_ENABLED setting.

    Auto-detects Jupyter environment and uses JupyterApprovalHandler if applicable.
    """
    if not constant.USER_APPROVAL_ENABLED:
        return AutoApprovalHandler()
    if _is_jupyter():
        logger.debug("Jupyter environment detected, using JupyterApprovalHandler")
        return JupyterApprovalHandler()
    return CLIApprovalHandler()


# Module-level singleton handler
_default_handler: ApprovalHandler | None = None


def set_handler(handler: ApprovalHandler) -> None:
    """Override the default approval handler (e.g. for Streamlit integration)."""
    global _default_handler
    _default_handler = handler


def _get_handler() -> ApprovalHandler:
    global _default_handler
    if _default_handler is None:
        _default_handler = get_default_handler()
    return _default_handler


def make_approval_node(
    node_name: str,
    summary_extractor: Callable[[Any], str],
    retry_target: str,
    next_target: str,
    on_retry: Callable[[Any], None] | None = None,
) -> tuple[Callable, Callable]:
    """Create an approval node function and its conditional edge function.

    Args:
        node_name: Display name for this approval checkpoint.
        summary_extractor: Function that extracts a summary string from agent state.
        retry_target: Node name to route to on reject/feedback (retry).
        next_target: Node name to route to on approval.
        on_retry: Optional callback invoked on rejection/feedback to reset state.

    Returns:
        A tuple of (node_function, conditional_edge_function).
    """

    def approval_node(agent_state: Any) -> Any:
        handler = _get_handler()
        summary = summary_extractor(agent_state)
        response = handler.request_approval(node_name, summary)

        if response.result == ApprovalResult.APPROVED:
            agent_state.approval_status = "approved"
        elif response.result == ApprovalResult.REJECTED:
            agent_state.approval_status = "retry"
            agent_state.add_message(
                Message(
                    role="user",
                    content=(
                        f"[User Feedback on {node_name}]\n"
                        "The user rejected the current output. "
                        "Please try again with a different approach."
                    ),
                    agent_sender="user_approval",
                ).with_log()
            )
            if on_retry is not None:
                on_retry(agent_state)
        elif response.result == ApprovalResult.FEEDBACK:
            agent_state.approval_status = "retry"
            agent_state.add_message(
                Message(
                    role="user",
                    content=(
                        f"[User Feedback on {node_name}]\n"
                        f"{response.feedback}\n\n"
                        "Please revise accordingly."
                    ),
                    agent_sender="user_approval",
                ).with_log()
            )
            if on_retry is not None:
                on_retry(agent_state)

        agent_state.intermediate_state.append(
            {
                "node_name": node_name,
                "output": f"Approval status: {agent_state.approval_status}",
            }
        )

        return agent_state

    def approval_conditional(agent_state: Any) -> str:
        if agent_state.approval_status == "approved":
            return next_target
        else:
            return retry_target

    return approval_node, approval_conditional
