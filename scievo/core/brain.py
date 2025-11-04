from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from secrets import randbelow
from threading import RLock

from dotenv import load_dotenv

load_dotenv()


class Brain:
    """Singleton container coordinating shared application state."""

    _instance: Brain | None = None
    _lock: RLock = RLock()

    def __new__(cls) -> Brain:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)

                    brain_dir = os.getenv("BRAIN_DIR")
                    if brain_dir is None:
                        raise ValueError("BRAIN_DIR environment variable must be set.")

                    cls._instance.brain_dir = Path(brain_dir)
                    cls._instance.brain_dir.mkdir(parents=True, exist_ok=True)
        return cls._instance

    @classmethod
    def instance(cls) -> Brain:
        """Accessor for the singleton instance, creating it on first use."""
        return cls()

    @classmethod
    def new_session_named(cls, session_name: str, session_prefix: str = "ss_") -> BrainSession:
        """Create a new session directory with a named session."""
        session_dir = cls.instance().brain_dir / f"{session_prefix}{session_name}"
        return BrainSession(session_dir)

    @classmethod
    def new_session(cls, session_prefix: str = "ss_") -> BrainSession:
        """Create a new session directory with a unique UTC timestamp + random suffix."""
        now = datetime.now()
        suffix = (
            f"{now.year:04d}{now.month:02d}{now.day:02d}-"
            f"{now.hour:02d}{now.minute:02d}{now.second:02d}_"
            f"{randbelow(1_000_000):06d}"
        )
        return cls.new_session_named(suffix, session_prefix=session_prefix)


class BrainSession:
    """Manages a single brain session with agent directory support."""

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._agent_dirs: dict[str, Path] = {}

    def add_agent_dir(self, agent_name: str) -> Path:
        """Add a subdirectory in the session dir for the given agent name.

        Args:
            agent_name: Name of the agent to create a directory for

        Returns:
            Path to the created agent directory
        """
        agent_dir = self.session_dir / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        self._agent_dirs[agent_name] = agent_dir
        return agent_dir

    @property
    def agent_dirs(self) -> dict[str, Path]:
        """Property to access agent directories dictionary."""
        return self._agent_dirs.copy()
