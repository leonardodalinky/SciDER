from pathlib import Path
from threading import RLock

from dotenv import load_dotenv

load_dotenv()


class Brain:
    """Singleton container coordinating shared application state."""

    _instance: "Brain" | None = None
    _lock: RLock = RLock()

    def __new__(cls) -> "Brain":  # noqa: D401 - standard singleton override
        """Create or return the shared Brain instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, brain_dir: str | Path) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.brain_dir = Path(brain_dir)
        self.brain_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def instance(cls) -> "Brain":
        """Convenience accessor for the singleton instance."""
        return cls()
