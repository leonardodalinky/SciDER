import io
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from typing import Any

from .manager import CommandContextManager, SessionManager


class CommandState(Enum):
    """State of a non-blocking command execution."""

    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


class CommandContextBase(ABC):
    """Base class for managing non-blocking command execution."""

    def __init__(self, session_id: str, command: str, timeout: float | None = None):
        self.session_id = session_id
        self.command = command
        self.timeout = timeout
        self.state = CommandState.RUNNING
        self.error: str | None = None
        self.start_buffer_position = 0
        self.end_buffer_position: int | None = None
        self._lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._stop_monitoring = threading.Event()

        # Context ID will be assigned when registered with CommandContextManager
        self.context_id: str | None = None

    @property
    def session(self) -> "SessionBase":
        """Get the session instance associated with this context."""
        s = SessionManager().get_session(self.session_id)
        if s is None:
            raise ValueError(f"Session with ID {self.session_id} not found")
        return s

    @abstractmethod
    def _monitor_completion(self):
        """Monitor thread that checks if command has completed. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _send_command(self):
        """Send the command to the session. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _cancel_command(self):
        """Cancel the running command. Must be implemented by subclasses."""
        pass

    def start(self):
        """Start the command execution and monitoring."""
        logger.debug(f"Starting non-blocking command: {self.command}")

        # Record history buffer position before sending command
        self.start_buffer_position = self.session.get_history_position()

        # Send the command (implementation specific)
        self._send_command()

        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_completion, daemon=True)
        self._monitor_thread.start()

        return self

    def is_running(self) -> bool:
        """Check if command is still running."""
        with self._lock:
            return self.state == CommandState.RUNNING

    def is_completed(self) -> bool:
        """Check if command completed successfully."""
        with self._lock:
            return self.state == CommandState.COMPLETED

    def has_error(self) -> bool:
        """Check if command encountered an error."""
        with self._lock:
            return self.state in (CommandState.ERROR, CommandState.TIMEOUT)

    def get_state(self) -> CommandState:
        """Get current state of the command."""
        with self._lock:
            return self.state

    def get_error(self) -> str | None:
        """Get error message if any."""
        with self._lock:
            return self.error

    @abstractmethod
    def get_input_output(self, max_length: int | None = None) -> str:
        """Get the input and output of the command. Used for AI conversation context."""
        pass

    def wait(self, timeout: float | None = None) -> bool:
        """
        Wait for command to complete.

        Args:
            timeout: Maximum time to wait in seconds. None means wait indefinitely.

        Returns:
            True if command completed, False if timeout occurred while waiting.
        """
        if self._monitor_thread:
            self._monitor_thread.join(timeout=timeout)
            return not self.is_running()
        return True

    def cancel(self):
        """Cancel the running command."""
        if self.is_running():
            logger.info(f"Cancelling command: {self.command}")
            self._stop_monitoring.set()
            self._cancel_command()
            with self._lock:
                self.state = CommandState.ERROR
                self.error = "Command cancelled by user"


class SessionBase(ABC):
    """Base class for managing shell sessions."""

    def __init__(self):
        # Create history buffer to record all I/O
        self.history_buffer = io.StringIO()

        # Track current command context
        self.current_context_id: str | None = None
        self._context_lock = threading.RLock()

        # Session ID will be assigned when registered with SessionManager
        self.session_id: str | None = None

    @abstractmethod
    def terminate_session(self):
        """Terminate the session. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def exec(self, command: str, timeout: float | None = None) -> CommandContextBase:
        """
        Execute command in non-blocking mode and return a context object.
        Must be implemented by subclasses.

        Args:
            command: The command to execute
            timeout: Maximum time to wait for command completion in seconds (default: None, no timeout)

        Returns:
            CommandContextBase: A context object that tracks the execution state
        """
        pass

    def get_history(self, start_position: int = 0) -> str:
        """
        Get the command history from the specified position.

        Args:
            start_position: Starting position in the history buffer (default: 0 for all history)

        Returns:
            The history content from the start position to current position
        """
        current_position = self.history_buffer.tell()
        self.history_buffer.seek(start_position)
        content = self.history_buffer.read()
        self.history_buffer.seek(current_position)
        return content

    def get_full_history(self) -> str:
        """
        Get the complete command history.

        Returns:
            All history content
        """
        return self.history_buffer.getvalue()

    def get_history_position(self) -> int:
        """
        Get the current position in the history buffer.

        Returns:
            Current position in the history buffer
        """
        return self.history_buffer.tell()

    def is_running_command(self) -> bool:
        """
        Check if a command is currently running in this session.

        Returns:
            True if a command is running, False otherwise.
        """
        with self._context_lock:
            return (
                self.get_current_context() is not None and self.get_current_context().is_running()
            )

    def get_current_context(self) -> CommandContextBase | None:
        """
        Get the current command context if one is running.

        Returns:
            The current CommandContextBase or None if no command is running.
        """
        with self._context_lock:
            return (
                CommandContextManager().get_context(self.current_context_id)
                if self.current_context_id
                else None
            )
