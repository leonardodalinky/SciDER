"""Streamlit-compatible approval handler.

Uses threading.Event to block the workflow background thread while the
Streamlit UI thread renders approval buttons and collects the user's response.
"""

import threading

from scider.core.approval import ApprovalHandler, ApprovalResponse, ApprovalResult


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

    # -- called from background (workflow) thread --

    def request_approval(self, node_name: str, summary: str) -> ApprovalResponse:
        with self._lock:
            self._pending = {"node_name": node_name, "summary": summary}
            self._event.clear()
            self._response = None
        self._event.wait()  # blocks until UI calls submit_response
        return self._response

    # -- called from UI (main) thread --

    def has_pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    def get_pending(self) -> dict | None:
        with self._lock:
            return self._pending

    def submit_response(self, response: ApprovalResponse) -> None:
        with self._lock:
            self._response = response
            self._pending = None
            self._event.set()
