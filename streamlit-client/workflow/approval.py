"""Streamlit-compatible approval handler.

Uses threading.Event to block the workflow background thread while the
Streamlit UI thread renders approval buttons and collects the user's response.
"""

import threading

from scider.core.approval import ApprovalHandler, ApprovalResponse, ApprovalResult
from workflow.runner import WorkflowCancelled


class StreamlitApprovalHandler(ApprovalHandler):
    """Approval handler for Streamlit UI.

    The workflow runs in a background thread. When an approval node fires,
    ``request_approval`` stores the pending request and blocks via
    ``threading.Event``. The Streamlit UI thread detects the pending request,
    renders buttons, and calls ``submit_response`` to unblock the workflow.
    """

    def __init__(self):
        self._event = threading.Event()
        self._response: ApprovalResponse | None = None
        self._pending: dict | None = None
        self._lock = threading.Lock()
        self._live_messages: list[dict] = []
        self._cancelled = False

    # -- called from background (workflow) thread --

    def request_approval(self, node_name: str, summary: str, title: str = "") -> ApprovalResponse:
        with self._lock:
            if self._cancelled:
                raise WorkflowCancelled()
            self._pending = {"node_name": node_name, "summary": summary, "title": title}
            self._event.clear()
            self._response = None
        self._event.wait()  # blocks until UI calls submit_response or cancel()
        if self._cancelled:
            raise WorkflowCancelled()
        return self._response

    # -- called from UI (main) thread --

    def has_pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    def get_pending(self) -> dict | None:
        with self._lock:
            return self._pending

    def request_approval_with_selection(
        self, node_name: str, summary: str, items: list[dict], title: str = ""
    ) -> ApprovalResponse:
        with self._lock:
            if self._cancelled:
                raise WorkflowCancelled()
            self._pending = {
                "node_name": node_name,
                "summary": summary,
                "title": title,
                "items": items,
                "has_selection": True,
            }
            self._event.clear()
            self._response = None
        self._event.wait()
        if self._cancelled:
            raise WorkflowCancelled()
        return self._response

    def cancel(self) -> None:
        """Unblock any pending approval wait; subsequent calls raise WorkflowCancelled."""
        with self._lock:
            self._cancelled = True
            self._pending = None
        self._event.set()

    def push_message(
        self,
        role: str,
        content: str,
        agent: str | None = None,
        is_meta: bool = False,
        is_tool: bool = False,
        tool_name: str | None = None,
        images: list[dict] | None = None,
    ) -> None:
        """Push a message for the UI thread to display (thread-safe).

        ``images`` is the tool-result image list — each entry is
        ``{"media_type": "image/png", "data": "<base64>"}`` — and is only
        set for tool results produced by image-aware tools like Read.
        """
        with self._lock:
            self._live_messages.append(
                {
                    "role": role,
                    "content": content,
                    "agent": agent,
                    "is_meta": is_meta,
                    "is_tool": is_tool,
                    "tool_name": tool_name,
                    "images": images,
                }
            )

    def drain_messages(self) -> list[dict]:
        """Drain all queued messages (called from UI thread)."""
        with self._lock:
            msgs = list(self._live_messages)
            self._live_messages.clear()
            return msgs

    def submit_response(self, response: ApprovalResponse) -> None:
        with self._lock:
            self._response = response
            self._pending = None
            self._event.set()
