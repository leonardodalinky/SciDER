"""
Workflow Monitor for Real-time Progress Tracking

This module provides callback hooks to monitor workflow progress in real-time.
"""

import queue
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Callable


class PhaseType(Enum):
    """Workflow phase types."""

    IDEATION_LITERATURE_SEARCH = "ideation_literature_search"
    IDEATION_ANALYZE_PAPERS = "ideation_analyze_papers"
    IDEATION_GENERATE_IDEAS = "ideation_generate_ideas"
    IDEATION_NOVELTY_CHECK = "ideation_novelty_check"
    IDEATION_REPORT = "ideation_report"

    DATA_PLANNING = "data_planning"
    DATA_EXECUTION = "data_execution"
    DATA_PAPER_SEARCH = "data_paper_search"
    DATA_FINALIZE = "data_finalize"

    EXPERIMENT_INIT = "experiment_init"
    EXPERIMENT_CODING = "experiment_coding"
    EXPERIMENT_EXEC = "experiment_exec"
    EXPERIMENT_SUMMARY = "experiment_summary"
    EXPERIMENT_ANALYSIS = "experiment_analysis"
    EXPERIMENT_REVISION = "experiment_revision"

    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ProgressUpdate:
    """A single progress update."""

    timestamp: float
    phase: PhaseType
    status: str  # "started", "progress", "completed", "error"
    message: str
    data: dict[str, Any] | None = None


class WorkflowMonitor:
    """Monitor workflow progress with real-time updates."""

    def __init__(self):
        self.updates: list[ProgressUpdate] = []
        self.update_queue: queue.Queue = queue.Queue()
        self.lock = Lock()
        self.callbacks: list[Callable[[ProgressUpdate], None]] = []

    def add_callback(self, callback: Callable[[ProgressUpdate], None]):
        """Add a callback function to be called on each update."""
        with self.lock:
            self.callbacks.append(callback)

    def log_update(
        self,
        phase: PhaseType,
        status: str,
        message: str,
        data: dict[str, Any] | None = None,
    ):
        """Log a progress update."""
        update = ProgressUpdate(
            timestamp=time.time(), phase=phase, status=status, message=message, data=data or {}
        )

        with self.lock:
            self.updates.append(update)
            self.update_queue.put(update)

            # Call all registered callbacks
            for callback in self.callbacks:
                try:
                    callback(update)
                except Exception as e:
                    print(f"Error in callback: {e}")

    def get_updates(self) -> list[ProgressUpdate]:
        """Get all updates."""
        with self.lock:
            return self.updates.copy()

    def get_latest_updates(self, count: int = 10) -> list[ProgressUpdate]:
        """Get the latest N updates."""
        with self.lock:
            return self.updates[-count:]

    def get_updates_by_phase(self, phase: PhaseType) -> list[ProgressUpdate]:
        """Get all updates for a specific phase."""
        with self.lock:
            return [u for u in self.updates if u.phase == phase]

    def clear(self):
        """Clear all updates."""
        with self.lock:
            self.updates.clear()
            # Clear the queue
            while not self.update_queue.empty():
                try:
                    self.update_queue.get_nowait()
                except queue.Empty:
                    break


# Global monitor instance
_global_monitor: WorkflowMonitor | None = None


def get_monitor() -> WorkflowMonitor:
    """Get the global workflow monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = WorkflowMonitor()
    return _global_monitor


def reset_monitor():
    """Reset the global monitor."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.clear()
    _global_monitor = WorkflowMonitor()
