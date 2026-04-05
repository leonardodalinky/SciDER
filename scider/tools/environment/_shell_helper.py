"""Shared shell helper for environment tools."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _detect_shell() -> str:
    if shutil.which("bash"):
        return "bash"
    return "sh"


_SHELL = _detect_shell()


def run_shell_cmd(command: str, cwd: str | None = None, timeout: int = 30) -> str:
    """Execute a shell command and return its output."""
    try:
        working_dir = None
        if cwd:
            working_dir = Path(os.path.expandvars(cwd)).expanduser()
            if not working_dir.exists():
                return f"Error: Working directory '{cwd}' does not exist"
            working_dir = str(working_dir)

        result = subprocess.run(
            [_SHELL, "-c", command],
            capture_output=True,
            text=True,
            cwd=working_dir,
            timeout=timeout,
        )

        output = []
        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")
        output.append(f"\nReturn code: {result.returncode}")

        return "\n".join(output) if output else "Command executed with no output"

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {e}"
