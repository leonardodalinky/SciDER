import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .registry import register_tool, register_toolset_desc

register_toolset_desc("shell", "Shell command execution toolset.")


def _detect_shell() -> str:
    """Detect available shell. Prefer bash if available, otherwise use sh."""
    if shutil.which("bash"):
        return "bash"
    return "sh"


# Detect the available shell at module load time
_SHELL = _detect_shell()


@register_tool(
    "shell",
    {
        "type": "function",
        "function": {
            "name": f"run_{_SHELL}_cmd",
            "description": f"Execute a {_SHELL} command and return its output (stdout and stderr combined).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": f"The {_SHELL} command to execute",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the command (optional)",
                        "default": None,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30)",
                        "default": 30,
                    },
                },
                "required": ["command"],
            },
        },
    },
)
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


@register_tool(
    "shell",
    {
        "type": "function",
        "function": {
            "name": f"run_{_SHELL}_script",
            "description": f"Execute a {_SHELL} script from a string and return its output (stdout and stderr combined).",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": f"The {_SHELL} script content to execute",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the script (optional)",
                        "default": None,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30)",
                        "default": 30,
                    },
                },
                "required": ["script"],
            },
        },
    },
)
def run_shell_script(script: str, cwd: str | None = None, timeout: int = 30) -> str:
    """Execute a shell script and return its output."""
    try:
        working_dir = None
        if cwd:
            working_dir = Path(os.path.expandvars(cwd)).expanduser()
            if not working_dir.exists():
                return f"Error: Working directory '{cwd}' does not exist"
            working_dir = str(working_dir)

        # Create a temporary script file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f".{_SHELL}",
            delete=False,
        ) as f:
            # Add shebang
            f.write(f"#!/usr/bin/{_SHELL}\n")
            f.write(script)
            script_path = f.name

        try:
            # Make the script executable
            os.chmod(script_path, 0o755)

            result = subprocess.run(
                [_SHELL, script_path],
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

            return "\n".join(output) if output else "Script executed with no output"

        finally:
            # Clean up the temporary script file
            try:
                os.unlink(script_path)
            except Exception:
                pass

    except subprocess.TimeoutExpired:
        return f"Error: Script timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing script: {e}"
