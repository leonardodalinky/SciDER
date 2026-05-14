"""Background workflow executor for Streamlit.

Runs a workflow function in a daemon thread so the Streamlit UI thread
remains responsive for rendering updates and handling approval interactions.
"""

import threading
import time
import traceback
from typing import Any, Callable


class WorkflowRunner:
    """Run a workflow function in a background daemon thread."""

    def __init__(self):
        self.result: Any = None
        self.error: Exception | None = None
        self.traceback: str | None = None
        self.is_running: bool = False
        self.is_done: bool = False
        self.start_time: float | None = None
        self._thread: threading.Thread | None = None

    def start(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        """Start *func* in a background thread."""
        self.result = None
        self.error = None
        self.traceback = None
        self.is_running = True
        self.is_done = False
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._run, args=(func, args, kwargs), daemon=True)
        self._thread.start()

    @property
    def elapsed(self) -> float:
        """Seconds since the workflow started (0 if not started)."""
        return (time.time() - self.start_time) if self.start_time else 0.0

    def _run(self, func: Callable, args: tuple, kwargs: dict) -> None:
        try:
            self.result = func(*args, **kwargs)
        except Exception as e:
            self.error = e
            self.traceback = traceback.format_exc()
        finally:
            self.is_running = False
            self.is_done = True
