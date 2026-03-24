"""Background workflow executor for Streamlit.

Runs a workflow function in a daemon thread so the Streamlit UI thread
remains responsive for rendering updates and handling approval interactions.
"""

import threading
from typing import Any, Callable


class WorkflowRunner:
    """Run a workflow function in a background daemon thread."""

    def __init__(self):
        self.result: Any = None
        self.error: Exception | None = None
        self.is_running: bool = False
        self.is_done: bool = False
        self._thread: threading.Thread | None = None

    def start(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        """Start *func* in a background thread."""
        self.result = None
        self.error = None
        self.is_running = True
        self.is_done = False
        self._thread = threading.Thread(target=self._run, args=(func, args, kwargs), daemon=True)
        self._thread.start()

    def _run(self, func: Callable, args: tuple, kwargs: dict) -> None:
        try:
            self.result = func(*args, **kwargs)
        except Exception as e:
            self.error = e
        finally:
            self.is_running = False
            self.is_done = True
