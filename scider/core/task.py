"""Task system for background execution.

Modeled after Claude Code's Task system (LocalShellTask + TaskOutput + TaskStop).
Provides background shell command execution with output capture to disk,
task lifecycle management, and notification injection into the query() loop.

Usage:
    # Spawn a background task (done by BashTool when run_in_background=True)
    task_id = TaskManager.spawn_shell(command="npm install", cwd="/workspace", timeout=300)

    # Check task status (done by TaskOutput tool)
    task = TaskManager.get(task_id)

    # Read task output (done by TaskOutput tool)
    output = TaskManager.read_output(task_id)

    # Stop a task (done by TaskStop tool)
    TaskManager.stop(task_id)

    # Drain notifications (done by query() before each LLM call)
    notifications = TaskManager.drain_notifications()
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock, RLock, Thread

from loguru import logger


class TaskStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class TaskType(str, Enum):
    SHELL = "shell"
    AGENT = "agent"


@dataclass
class TaskState:
    """State of a background task."""

    id: str
    type: TaskType
    status: TaskStatus
    command: str
    description: str
    output_path: str  # file path for stdout+stderr
    cwd: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    exit_code: int | None = None
    notified: bool = False

    # Internal — not exposed to tools
    _process: subprocess.Popen | None = field(default=None, repr=False)
    _thread: Thread | None = field(default=None, repr=False)


# Max output file size before killing the task (50 MB)
MAX_OUTPUT_BYTES = 50 * 1024 * 1024


def _generate_task_id() -> str:
    return f"task_{uuid.uuid4().hex[:8]}"


def _get_output_dir() -> Path:
    """Get or create the task output directory."""
    d = Path(tempfile.gettempdir()) / "scider_tasks"
    d.mkdir(parents=True, exist_ok=True)
    return d


class TaskManager:
    """Singleton manager for background tasks.

    Thread-safe. All mutations go through the lock.
    """

    _instance: TaskManager | None = None
    _lock: RLock = RLock()

    def __new__(cls) -> TaskManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._tasks: dict[str, TaskState] = {}
        self._notifications: list[str] = []
        self._notify_lock = Lock()

    @classmethod
    def instance(cls) -> TaskManager:
        return cls()

    # --- Spawn ---

    @classmethod
    def spawn_shell(
        cls,
        *,
        command: str,
        cwd: str | None = None,
        timeout: int = 600,
        description: str | None = None,
    ) -> str:
        """Spawn a shell command in the background. Returns task_id."""
        mgr = cls.instance()
        task_id = _generate_task_id()
        output_path = str(_get_output_dir() / f"{task_id}.output")

        task = TaskState(
            id=task_id,
            type=TaskType.SHELL,
            status=TaskStatus.RUNNING,
            command=command,
            description=description or command[:100],
            output_path=output_path,
            cwd=cwd,
        )

        with cls._lock:
            mgr._tasks[task_id] = task

        # Start background thread
        thread = Thread(
            target=cls._run_shell,
            args=(task_id, command, cwd, timeout, output_path),
            daemon=True,
            name=f"task-{task_id}",
        )
        task._thread = thread
        thread.start()

        logger.info("Spawned background task {}: {}", task_id, command[:80])
        return task_id

    @classmethod
    def _run_shell(
        cls,
        task_id: str,
        command: str,
        cwd: str | None,
        timeout: int,
        output_path: str,
    ) -> None:
        """Execute shell command in background thread, writing output to file."""
        mgr = cls.instance()
        process = None
        try:
            with open(output_path, "w") as f:
                process = subprocess.Popen(
                    ["bash", "-c", command],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=cwd,
                    text=True,
                )

            # Store process reference for potential kill
            with cls._lock:
                task = mgr._tasks.get(task_id)
                if task:
                    task._process = process

            # Wait with timeout + size watchdog
            start = time.time()
            while process.poll() is None:
                elapsed = time.time() - start
                if elapsed > timeout:
                    process.kill()
                    process.wait(timeout=5)
                    cls._complete_task(task_id, TaskStatus.KILLED, -1)
                    logger.warning("Task {} killed: timeout after {}s", task_id, timeout)
                    return

                # Size watchdog
                try:
                    size = os.path.getsize(output_path)
                    if size > MAX_OUTPUT_BYTES:
                        process.kill()
                        process.wait(timeout=5)
                        cls._complete_task(task_id, TaskStatus.KILLED, -1)
                        logger.warning(
                            "Task {} killed: output exceeded {} bytes", task_id, MAX_OUTPUT_BYTES
                        )
                        return
                except OSError:
                    pass

                time.sleep(1)

            # Process completed naturally
            exit_code = process.returncode
            status = TaskStatus.COMPLETED if exit_code == 0 else TaskStatus.FAILED
            cls._complete_task(task_id, status, exit_code)

        except Exception as e:
            logger.exception("Task {} crashed: {}", task_id, e)
            cls._complete_task(task_id, TaskStatus.FAILED, -1)

    @classmethod
    def _complete_task(cls, task_id: str, status: TaskStatus, exit_code: int) -> None:
        """Mark task as completed and enqueue notification."""
        mgr = cls.instance()
        with cls._lock:
            task = mgr._tasks.get(task_id)
            if not task:
                return
            task.status = status
            task.exit_code = exit_code
            task.end_time = time.time()
            task._process = None

        # Build notification
        elapsed = (task.end_time or 0) - task.start_time
        notification = (
            f"<task-notification>\n"
            f"  <task-id>{task_id}</task-id>\n"
            f"  <status>{status.value}</status>\n"
            f"  <exit-code>{exit_code}</exit-code>\n"
            f"  <elapsed>{elapsed:.1f}s</elapsed>\n"
            f'  <summary>Background command "{task.description}" {status.value}'
            f'{f" (exit {exit_code})" if exit_code is not None else ""}</summary>\n'
            f"  <output-file>{task.output_path}</output-file>\n"
            f"</task-notification>"
        )

        with mgr._notify_lock:
            mgr._notifications.append(notification)

        logger.info(
            "Task {} {}: exit_code={}, elapsed={:.1f}s", task_id, status.value, exit_code, elapsed
        )

    # --- Query ---

    @classmethod
    def get(cls, task_id: str) -> TaskState | None:
        """Get task state by ID."""
        mgr = cls.instance()
        with cls._lock:
            return mgr._tasks.get(task_id)

    @classmethod
    def list_tasks(cls) -> list[TaskState]:
        """List all tasks."""
        mgr = cls.instance()
        with cls._lock:
            return list(mgr._tasks.values())

    @classmethod
    def read_output(cls, task_id: str, tail: int | None = None) -> str:
        """Read task output from disk.

        Args:
            task_id: Task identifier.
            tail: If set, return only the last N lines.
        """
        mgr = cls.instance()
        with cls._lock:
            task = mgr._tasks.get(task_id)
        if not task:
            return f"Error: Task {task_id} not found"

        try:
            with open(task.output_path, "r", errors="replace") as f:
                content = f.read()
            if tail is not None and tail > 0:
                lines = content.splitlines()
                content = "\n".join(lines[-tail:])
            return content
        except FileNotFoundError:
            return "No output available yet"
        except Exception as e:
            return f"Error reading output: {e}"

    # --- Control ---

    @classmethod
    def stop(cls, task_id: str) -> str:
        """Stop a running task. Returns status message."""
        mgr = cls.instance()
        with cls._lock:
            task = mgr._tasks.get(task_id)

        if not task:
            return f"Error: Task {task_id} not found"
        if task.status != TaskStatus.RUNNING:
            return f"Task {task_id} is not running (status: {task.status.value})"

        process = task._process
        if process:
            try:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            except Exception as e:
                logger.warning("Error stopping task {}: {}", task_id, e)

        cls._complete_task(task_id, TaskStatus.KILLED, -1)
        return f"Task {task_id} stopped"

    # --- Notifications ---

    @classmethod
    def drain_notifications(cls) -> list[str]:
        """Drain and return all pending task notifications.

        Called by query() before each LLM call to inject notifications
        as system-reminder messages.
        """
        mgr = cls.instance()
        with mgr._notify_lock:
            notifications = mgr._notifications.copy()
            mgr._notifications.clear()
        return notifications

    @classmethod
    def wait_for_task(cls, task_id: str, timeout: float = 30.0) -> TaskState | None:
        """Block until task completes or timeout. Returns final task state."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            task = cls.get(task_id)
            if task and task.status != TaskStatus.RUNNING:
                return task
            time.sleep(0.5)
        return cls.get(task_id)
