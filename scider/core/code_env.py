import os
import shutil
import subprocess
from contextlib import AbstractContextManager
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, PrivateAttr


class LocalEnv(AbstractContextManager, BaseModel):
    """Context manager that temporarily switches the working directory."""

    working_dir: Path
    _original_cwd: Path | None = PrivateAttr(default=None)

    @logger.catch
    def __init__(self, working_dir: str | Path, create_dir_if_missing: bool = True):
        """Initialise the environment with an optional auto-create directory flag."""
        # Resolve and validate the target directory.
        working_dir = Path(working_dir).resolve()
        super().__init__(working_dir=working_dir, create_dir_if_missing=create_dir_if_missing)
        if self.working_dir.exists():
            if not self.working_dir.is_dir():
                raise NotADirectoryError(f"Path {self.working_dir} is not a directory")
        else:
            if create_dir_if_missing:
                self.working_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Directory {self.working_dir} does not exist")

        # If uv is available and the workspace has no pyproject.toml yet,
        # initialise an isolated Python project so that `uv run` / `uv add`
        # operate on the workspace's own venv instead of SciDER's.
        self._ensure_uv_project()

    def _ensure_uv_project(self) -> None:
        """If uv is available and the workspace lacks pyproject.toml, run `uv init`."""
        if not shutil.which("uv"):
            return
        if (self.working_dir / "pyproject.toml").exists():
            return
        try:
            result = subprocess.run(
                ["uv", "init", "--no-readme", "--no-pin-python", "--vcs", "none"],
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                # Remove default boilerplate files that uv init creates
                for name in ("hello.py", "main.py"):
                    p = self.working_dir / name
                    if p.exists():
                        p.unlink()
                logger.debug("uv init: created isolated project in {}", self.working_dir)
            else:
                logger.debug("uv init failed (rc={}): {}", result.returncode, result.stderr.strip())
        except Exception as e:
            logger.debug("uv init skipped: {}", e)

    def __enter__(self) -> "LocalEnv":
        """Switch into the directory, and return the context."""
        # Ensure directory exists and move into it.
        self._original_cwd = Path.cwd()
        os.chdir(self.working_dir)
        logger.trace("Switched to directory: {}", self.working_dir)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Restore the previous working directory when the context ends."""
        # Restore original working directory after the block ends.
        if self._original_cwd is not None:
            os.chdir(self._original_cwd)
            self._original_cwd = None
        logger.trace("Switched back to directory: {}", self.working_dir)
        return False
