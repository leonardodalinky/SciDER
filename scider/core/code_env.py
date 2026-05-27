import os
import shutil
import subprocess
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr


class WorkspaceInitConfig(BaseModel):
    """How a workspace should be initialised and how agents should talk to it.

    Three orthogonal knobs:

    - `env_manager`: which tool agents are told to use in system context
      ("uv" → `uv run python`, `uv add ...`; "python" → bare `python`, `pip install`).
    - `init_uv`: whether `LocalEnv` runs `uv init` on a fresh workspace.
    - `venv_path`: if set, `<venv_path>/bin` is prepended to `PATH` while the
      `LocalEnv` is active, so `python` / installed scripts resolve to that venv.

    Defaults preserve the pre-existing behaviour (uv-managed workspaces).
    """

    env_manager: Literal["python", "uv"] = "uv"
    init_uv: bool = True
    venv_path: Path | None = None

    model_config = {"frozen": True}

    def cache_key(self) -> tuple:
        """Hashable key for per-config caches (e.g. runtime-detection cache)."""
        return (self.env_manager, self.init_uv, str(self.venv_path) if self.venv_path else None)


# Singleton default — safe because WorkspaceInitConfig is frozen.
DEFAULT_WORKSPACE_INIT_CONFIG = WorkspaceInitConfig()


class LocalEnv(AbstractContextManager, BaseModel):
    """Context manager that temporarily switches the working directory.

    Optionally initialises an isolated uv project, and optionally prepends a
    venv's `bin/` to `PATH` while the context is active. Both behaviours are
    driven by `init_config` (see `WorkspaceInitConfig`). Leaving `init_config`
    unset reproduces the historical behaviour: uv-managed workspaces, auto
    `uv init`, no PATH injection.
    """

    working_dir: Path
    init_config: WorkspaceInitConfig = Field(default_factory=lambda: DEFAULT_WORKSPACE_INIT_CONFIG)
    _original_cwd: Path | None = PrivateAttr(default=None)
    _original_path: str | None = PrivateAttr(default=None)

    @logger.catch
    def __init__(
        self,
        working_dir: str | Path,
        create_dir_if_missing: bool = True,
        init_config: WorkspaceInitConfig | None = None,
    ):
        """Initialise the environment with an optional auto-create directory flag."""
        # Resolve and validate the target directory.
        working_dir = Path(working_dir).resolve()
        super().__init__(
            working_dir=working_dir,
            create_dir_if_missing=create_dir_if_missing,
            init_config=init_config or DEFAULT_WORKSPACE_INIT_CONFIG,
        )
        if self.working_dir.exists():
            if not self.working_dir.is_dir():
                raise NotADirectoryError(f"Path {self.working_dir} is not a directory")
        else:
            if create_dir_if_missing:
                self.working_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Directory {self.working_dir} does not exist")

        # Only run `uv init` when the config asks for a uv-managed workspace.
        if self.init_config.env_manager == "uv" and self.init_config.init_uv:
            self._ensure_uv_project()

    def _ensure_uv_project(self) -> None:
        """If uv is available and the workspace lacks pyproject.toml, run `uv init`."""
        if not shutil.which("uv"):
            return
        if (self.working_dir / "pyproject.toml").exists():
            return
        try:
            result = subprocess.run(
                # --no-workspace keeps this an isolated project: without it, a
                # workspace created under the repo attaches to the root
                # pyproject.toml's [tool.uv.workspace] members and pollutes it.
                ["uv", "init", "--no-readme", "--no-pin-python", "--vcs", "none",
                 "--no-workspace"],
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
        """Switch into the directory, optionally inject venv/bin into PATH."""
        self._original_cwd = Path.cwd()
        os.chdir(self.working_dir)
        logger.trace("Switched to directory: {}", self.working_dir)

        if self.init_config.venv_path is not None:
            venv_bin = Path(self.init_config.venv_path).resolve() / "bin"
            self._original_path = os.environ.get("PATH", "")
            # Prepend, preserving existing entries.
            os.environ["PATH"] = f"{venv_bin}{os.pathsep}{self._original_path}"
            logger.trace("Prepended {} to PATH", venv_bin)

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Restore the previous working directory (and PATH) when the context ends."""
        if self._original_cwd is not None:
            os.chdir(self._original_cwd)
            self._original_cwd = None
        if self._original_path is not None:
            os.environ["PATH"] = self._original_path
            self._original_path = None
        logger.trace("Switched back to directory: {}", self.working_dir)
        return False
