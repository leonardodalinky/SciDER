"""User approval mechanism for agent workflows.

Provides interactive checkpoints where users can approve, reject, or provide
feedback at critical agent steps. Controlled by USER_APPROVAL_ENABLED env var.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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

    def __init__(
        self,
        result: ApprovalResult,
        feedback: str | None = None,
        selected_index: int | None = None,
    ):
        self.result = result
        self.feedback = feedback
        self.selected_index = selected_index  # For single-item selection approvals


@dataclass
class ApprovalContext:
    """Context for result-style approvals — passed to subagent-capable handlers.

    Only populated by call sites that want a potential LLM approval subagent
    to have access to the parent agent's artifacts. Other handlers ignore it.
    """

    parent_state: Any | None = None  # HistoryState subclass
    parent_agent: str = ""  # "data" | "experiment" | ...
    workspace_dir: Path | None = None
    # Parent's LocalEnv — forwarded so the approval subagent's Bash/Read
    # tool calls run with cwd = parent workspace (and the right PATH).
    # Without it the subagent inherits the driver's cwd (project root).
    workspace: Any | None = None
    critic_feedback: str | None = None
    user_query: str = ""


class ApprovalHandler(ABC):
    """Abstract base class for approval handlers."""

    @abstractmethod
    def request_approval(
        self,
        node_name: str,
        summary: str,
        title: str = "",
        *,
        context: ApprovalContext | None = None,
    ) -> ApprovalResponse:
        """Request user approval. Blocks until user responds.

        Args:
            node_name: Name of the node that produced the output.
            summary: Human-readable summary of the output to review.
            title: Short abstract/title describing what this approval is about.
            context: Optional parent-state context for subagent-capable handlers.

        Returns:
            ApprovalResponse with the user's decision and optional feedback.
        """
        ...

    def request_approval_with_selection(
        self, node_name: str, summary: str, items: list[dict], title: str = ""
    ) -> ApprovalResponse:
        """Request approval with single-item selection.

        Default implementation formats items into the summary and delegates to
        ``request_approval``.  Subclasses may override for richer UI
        (radio buttons, numbered input, etc.).

        Args:
            items: Selectable items, each with at least a ``title`` key.
        """
        item_text = "\n".join(
            f"  [{i + 1}] {it.get('title', 'Untitled')}" for i, it in enumerate(items)
        )
        full_summary = f"{summary}\n\nItems:\n{item_text}"
        return self.request_approval(node_name, full_summary, title=title)


class CLIApprovalHandler(ApprovalHandler):
    """CLI-based approval handler using input() for blocking user interaction."""

    def request_approval(
        self,
        node_name: str,
        summary: str,
        title: str = "",
        *,
        context: ApprovalContext | None = None,
    ) -> ApprovalResponse:
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"  User Approval Required: [{node_name}]")
        if title:
            print(f"  {title}")
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

    def request_approval_with_selection(
        self, node_name: str, summary: str, items: list[dict], title: str = ""
    ) -> ApprovalResponse:
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"  Select & Approve: [{node_name}]")
        if title:
            print(f"  {title}")
        print(separator)
        print(summary)
        print(separator)
        for i, it in enumerate(items):
            title = it.get("title", "Untitled")
            desc = it.get("description", "")
            print(f"  [{i + 1}] {title}")
            if desc:
                print(f"      {desc[:120]}")
        print(separator)
        print(f"  Enter 1-{len(items)} to select and approve")
        print("  [r] Reject and retry")
        print("  [f] Provide feedback and retry")
        print(separator)

        while True:
            choice = input(f"Your choice [1-{len(items)}/r/f] (enter = 1): ").strip()
            if choice == "":
                choice = "1"
            if choice.lower() == "r":
                logger.info("User rejected [{}]", node_name)
                return ApprovalResponse(result=ApprovalResult.REJECTED)
            elif choice.lower() == "f":
                feedback = input("Your feedback (enter = cancel): ").strip()
                if not feedback:
                    print("Feedback cancelled, returning to choices.")
                    continue
                logger.info("User provided feedback for [{}]: {}", node_name, feedback)
                return ApprovalResponse(result=ApprovalResult.FEEDBACK, feedback=feedback)
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(items):
                    logger.info(
                        "User selected item {} for [{}]: {}",
                        idx,
                        node_name,
                        items[idx].get("title", ""),
                    )
                    return ApprovalResponse(result=ApprovalResult.APPROVED, selected_index=idx)
                else:
                    print(f"Invalid number. Please enter 1-{len(items)}.")
            else:
                print(f"Invalid choice. Enter 1-{len(items)}, r, or f.")


class JupyterApprovalHandler(ApprovalHandler):
    """Jupyter notebook approval handler with rich display."""

    def request_approval(
        self,
        node_name: str,
        summary: str,
        title: str = "",
        *,
        context: ApprovalContext | None = None,
    ) -> ApprovalResponse:
        from IPython.display import Markdown, display

        title_line = f"\n**{title}**\n" if title else ""
        display(
            Markdown(
                f"---\n### User Approval Required: `{node_name}`\n{title_line}\n{summary}\n\n---"
            )
        )
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

    def request_approval_with_selection(
        self, node_name: str, summary: str, items: list[dict], title: str = ""
    ) -> ApprovalResponse:
        from IPython.display import Markdown, display

        title_line = f"\n**{title}**\n" if title else ""
        lines = [f"---\n### Select & Approve: `{node_name}`\n{title_line}\n{summary}\n"]
        for i, it in enumerate(items):
            title = it.get("title", "Untitled")
            desc = it.get("description", "")
            lines.append(f"**[{i + 1}]** {title}")
            if desc:
                lines.append(f"> {desc[:200]}\n")
        lines.append("---")
        display(Markdown("\n".join(lines)))

        print(f"Enter 1-{len(items)} to select, [r] reject, [f] feedback")
        while True:
            choice = input(f"Your choice [1-{len(items)}/r/f] (enter = 1): ").strip()
            if choice == "":
                choice = "1"
            if choice.lower() == "r":
                return ApprovalResponse(result=ApprovalResult.REJECTED)
            elif choice.lower() == "f":
                feedback = input("Your feedback (enter = cancel): ").strip()
                if not feedback:
                    print("Feedback cancelled.")
                    continue
                return ApprovalResponse(result=ApprovalResult.FEEDBACK, feedback=feedback)
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(items):
                    logger.info("User selected item {} for [{}]", idx, node_name)
                    return ApprovalResponse(result=ApprovalResult.APPROVED, selected_index=idx)
                else:
                    print(f"Invalid. Enter 1-{len(items)}.")
            else:
                print(f"Invalid. Enter 1-{len(items)}, r, or f.")


class AutoApprovalHandler(ApprovalHandler):
    """Auto-approval handler. Always approves without user interaction."""

    def request_approval(
        self,
        node_name: str,
        summary: str,
        title: str = "",
        *,
        context: ApprovalContext | None = None,
    ) -> ApprovalResponse:
        logger.debug("Auto-approved [{}]", node_name)
        return ApprovalResponse(result=ApprovalResult.APPROVED)

    def request_approval_with_selection(
        self, node_name: str, summary: str, items: list[dict], title: str = ""
    ) -> ApprovalResponse:
        logger.debug("Auto-approved [{}], selected first item", node_name)
        return ApprovalResponse(
            result=ApprovalResult.APPROVED,
            selected_index=0 if items else None,
        )


class SubagentApprovalHandler(ApprovalHandler):
    """Invoke an LLM approval subagent for whitelisted nodes,
    auto-approve everything else.

    Currently only ``user_review`` (critic's result approval) is hooked.
    Non-hookable nodes and selection-style approvals fall through to
    blind APPROVED (matches AutoApprovalHandler semantics), which keeps
    backwards compatibility for ``AskUserQuestion`` and plan-style approvals.
    """

    HOOKED_NODES: set[str] = {"user_review"}

    def request_approval(
        self,
        node_name: str,
        summary: str,
        title: str = "",
        *,
        context: ApprovalContext | None = None,
    ) -> ApprovalResponse:
        if node_name not in self.HOOKED_NODES or context is None:
            logger.debug("SubagentApprovalHandler: passthrough [{}]", node_name)
            return ApprovalResponse(result=ApprovalResult.APPROVED)

        # Lazy import to avoid circular dependency (approval <- subagent <- query).
        from scider.agents.approval_subagent import run_approval_subagent

        try:
            verdict, feedback = run_approval_subagent(
                node_name=node_name,
                summary=summary,
                title=title,
                context=context,
            )
        except Exception as e:
            logger.warning("Approval subagent failed ({}), fail-open APPROVED", e)
            return ApprovalResponse(result=ApprovalResult.APPROVED)

        logger.info("Approval subagent verdict [{}]: {}", node_name, verdict)
        if verdict == "approve":
            return ApprovalResponse(result=ApprovalResult.APPROVED)
        return ApprovalResponse(
            result=ApprovalResult.FEEDBACK,
            feedback=feedback or "Approval subagent requested revisions.",
        )

    def request_approval_with_selection(
        self, node_name: str, summary: str, items: list[dict], title: str = ""
    ) -> ApprovalResponse:
        # Selection-style approvals (AskUserQuestion) are never hooked in this
        # release. Preserve AutoApprovalHandler semantics: APPROVED + first item.
        logger.debug(
            "SubagentApprovalHandler: selection passthrough [{}], picking first item",
            node_name,
        )
        return ApprovalResponse(
            result=ApprovalResult.APPROVED,
            selected_index=0 if items else None,
        )


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
    When auto-approval is on, uses SubagentApprovalHandler (LLM judge for
    user_review) if APPROVAL_SUBAGENT_ENABLED, else AutoApprovalHandler.
    """
    if not constant.USER_APPROVAL_ENABLED:
        if constant.APPROVAL_SUBAGENT_ENABLED:
            logger.debug(
                "Auto-approval on + APPROVAL_SUBAGENT_ENABLED — using SubagentApprovalHandler"
            )
            return SubagentApprovalHandler()
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
    title: str = "",
) -> tuple[Callable, Callable]:
    """Create an approval node function and its conditional edge function.

    Args:
        node_name: Display name for this approval checkpoint.
        summary_extractor: Function that extracts a summary string from agent state.
        retry_target: Node name to route to on reject/feedback (retry).
        next_target: Node name to route to on approval.
        on_retry: Optional callback invoked on rejection/feedback to reset state.
        title: Short abstract describing what this approval is about.

    Returns:
        A tuple of (node_function, conditional_edge_function).
    """

    def approval_node(agent_state: Any) -> Any:
        handler = _get_handler()
        summary = summary_extractor(agent_state)
        response = handler.request_approval(node_name, summary, title=title)

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


def make_selection_approval_node(
    node_name: str,
    summary_extractor: Callable[[Any], str],
    items_extractor: Callable[[Any], list[dict]],
    selection_handler: Callable[[Any, int], None],
    retry_target: str,
    next_target: str,
    on_retry: Callable[[Any], None] | None = None,
    title: str = "",
) -> tuple[Callable, Callable]:
    """Create an approval node with single-item selection.

    Like ``make_approval_node`` but additionally presents selectable items.
    If items list is empty, falls back to a regular approval prompt.

    Args:
        items_extractor: Extracts selectable items (list[dict]) from state.
        selection_handler: Called with ``(state, selected_index)`` on approval.
        title: Short abstract describing what this approval is about.
    """

    def selection_node(agent_state: Any) -> Any:
        handler = _get_handler()
        summary = summary_extractor(agent_state)
        items = items_extractor(agent_state)

        if items:
            response = handler.request_approval_with_selection(
                node_name, summary, items, title=title
            )
        else:
            response = handler.request_approval(node_name, summary, title=title)

        if response.result == ApprovalResult.APPROVED:
            agent_state.approval_status = "approved"
            if response.selected_index is not None:
                selection_handler(agent_state, response.selected_index)
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
                "output": f"Approval status: {agent_state.approval_status}"
                + (
                    f", selected: {response.selected_index}"
                    if response.selected_index is not None
                    else ""
                ),
            }
        )

        return agent_state

    def selection_conditional(agent_state: Any) -> str:
        if agent_state.approval_status == "approved":
            return next_target
        else:
            return retry_target

    return selection_node, selection_conditional
