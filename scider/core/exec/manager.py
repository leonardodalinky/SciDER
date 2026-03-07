import threading
import uuid
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .session_base import CommandContextBase, SessionBase


class SessionManager:
    """Singleton registry manager for SessionBase instances."""

    _instance: "SessionManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "SessionManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._sessions: dict[str, "SessionBase"] = {}
        self._session_lock = threading.Lock()
        self._initialized = True
        logger.debug("SessionManager initialized")

    def register_session(self, session: "SessionBase") -> str:
        """
        Register a session and return its unique ID. The session ID is also set on the session instance.

        Args:
            session: The SessionBase instance to register

        Returns:
            str: Unique session ID
        """
        if session.session_id is not None:
            raise ValueError("Session is already registered with an ID")
        session_id = "sess_" + str(uuid.uuid4())
        with self._session_lock:
            self._sessions[session_id] = session
        session.session_id = session_id
        logger.debug(f"Session registered with ID: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> "SessionBase | None":
        """
        Get a session by its ID.

        Args:
            session_id: The session ID

        Returns:
            SessionBase instance or None if not found
        """
        with self._session_lock:
            return self._sessions.get(session_id)

    def unregister_session(self, session_id: str) -> bool:
        """
        Unregister a session and terminate it.

        Args:
            session_id: The session ID

        Returns:
            bool: True if session was found and unregistered, False otherwise
        """
        with self._session_lock:
            session = self._sessions.pop(session_id, None)
        if session:
            session.terminate_session()
            logger.debug(f"Session unregistered: {session_id}")
            return True
        return False

    def list_sessions(self) -> list[str]:
        """
        Get list of all registered session IDs.

        Returns:
            list[str]: List of session IDs
        """
        with self._session_lock:
            return list(self._sessions.keys())

    def get_all_sessions(self) -> dict[str, "SessionBase"]:
        """
        Get all registered sessions.

        Returns:
            dict[str, SessionBase]: Dictionary of session ID to SessionBase instance
        """
        with self._session_lock:
            return dict(self._sessions)

    def clear_all_sessions(self):
        """Terminate and clear all sessions."""
        with self._session_lock:
            for session in self._sessions.values():
                try:
                    session.terminate_session()
                except Exception as e:
                    logger.error(f"Error terminating session: {e}")
            self._sessions.clear()
        logger.debug("All sessions cleared")


class CommandContextManager:
    """Singleton registry manager for CommandContextBase instances."""

    _instance: "CommandContextManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "CommandContextManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._contexts: dict[str, "CommandContextBase"] = {}
        self._context_lock = threading.Lock()
        self._initialized = True
        logger.debug("CommandContextManager initialized")

    def register_context(self, context: "CommandContextBase") -> str:
        """
        Register a command context and return its unique ID. The context ID is also set on the context instance.

        Args:
            context: The CommandContextBase instance to register

        Returns:
            str: Unique context ID
        """
        if context.context_id is not None:
            raise ValueError("Context is already registered with an ID")
        context_id = "ctx_" + str(uuid.uuid4())
        with self._context_lock:
            self._contexts[context_id] = context
        context.context_id = context_id
        logger.debug(f"Command context registered with ID: {context_id}")
        return context_id

    def get_context(self, context_id: str) -> "CommandContextBase | None":
        """
        Get a command context by its ID.

        Args:
            context_id: The context ID

        Returns:
            CommandContextBase instance or None if not found
        """
        with self._context_lock:
            return self._contexts.get(context_id)

    def unregister_context(self, context_id: str) -> bool:
        """
        Unregister a command context.

        Args:
            context_id: The context ID

        Returns:
            bool: True if context was found and unregistered, False otherwise
        """
        with self._context_lock:
            context = self._contexts.pop(context_id, None)
        if context:
            logger.debug(f"Command context unregistered: {context_id}")
            return True
        return False

    def list_contexts(self) -> list[str]:
        """
        Get list of all registered context IDs.

        Returns:
            list[str]: List of context IDs
        """
        with self._context_lock:
            return list(self._contexts.keys())

    def get_all_contexts(self) -> dict[str, "CommandContextBase"]:
        """
        Get all registered contexts.

        Returns:
            dict[str, CommandContextBase]: Dictionary of context ID to CommandContextBase instance
        """
        with self._context_lock:
            return dict(self._contexts)

    def clear_all_contexts(self):
        """Cancel and clear all contexts."""
        with self._context_lock:
            for context in self._contexts.values():
                try:
                    if context.is_running():
                        context.cancel()
                except Exception as e:
                    logger.error(f"Error cancelling context: {e}")
            self._contexts.clear()
        logger.debug("All command contexts cleared")
