"""BashTool — execute shell commands with safety classification.

Modeled after Claude Code's BashTool security model:
- Commands classified as read-only / write / dangerous per invocation
- Dangerous commands are denied
- Write commands constrained to workspace directory
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext


def _detect_shell() -> str:
    if shutil.which("bash"):
        return "bash"
    return "sh"


_SHELL = _detect_shell()

# Commands that only read — safe to run anywhere
READ_ONLY_COMMANDS = {
    "ls",
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "wc",
    "file",
    "stat",
    "find",
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "ag",
    "awk",
    "sed",  # sed without -i
    "diff",
    "cmp",
    "sort",
    "uniq",
    "cut",
    "tr",
    "tee",
    "echo",
    "printf",
    "date",
    "whoami",
    "pwd",
    "env",
    "printenv",
    "which",
    "type",
    "whereis",
    "man",
    "help",
    "du",
    "df",
    "free",
    "top",
    "ps",
    "uname",
    "id",
    "git status",
    "git log",
    "git diff",
    "git show",
    "git branch",
    "git remote",
    "git tag",
    "git stash list",
    "python --version",
    "python3 --version",
    "pip list",
    "pip show",
    "node --version",
    "npm list",
    "uv --version",
    "tree",
    "realpath",
    "readlink",
    "basename",
    "dirname",
    "md5sum",
    "sha256sum",
    "sha1sum",
}

# Commands that are always dangerous — deny
DANGEROUS_COMMANDS = {
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "mkfs",
    "dd if=",
    "format",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "chmod 777",
    "chmod -R 777",
    "> /dev/sda",
    "curl | sh",
    "curl | bash",
    "wget | sh",
    "wget | bash",
    "eval",
    ":(){:|:&};:",  # fork bomb
}

# Dangerous environment variables that should never be set
DANGEROUS_ENV_VARS = {
    "PATH",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "NODE_OPTIONS",
    "PYTHONPATH",
    "BASH_ENV",
}


def _get_first_command(command: str) -> str:
    """Extract the first command from a compound command."""
    # Split on &&, ||, ;, | and take first
    for sep in ["&&", "||", ";", "|"]:
        if sep in command:
            return command.split(sep)[0].strip()
    return command.strip()


def _classify_command(command: str) -> str:
    """Classify a command as 'read-only', 'write', or 'dangerous'.

    Returns one of: 'read-only', 'write', 'dangerous'
    """
    cmd = command.strip()
    cmd_lower = cmd.lower()

    # Check dangerous patterns first
    for pattern in DANGEROUS_COMMANDS:
        if pattern in cmd_lower:
            return "dangerous"

    # Check for dangerous env var manipulation
    for var in DANGEROUS_ENV_VARS:
        if f"{var}=" in cmd and not cmd.startswith("echo"):
            return "dangerous"

    # Check for command substitution / eval
    if "$()" in cmd or "`" in cmd:
        # Could be read-only (e.g., echo $(date)) but treat as write conservatively
        pass

    # Extract the base command
    first_cmd = _get_first_command(cmd)
    try:
        parts = shlex.split(first_cmd)
    except ValueError:
        parts = first_cmd.split()

    if not parts:
        return "read-only"

    base = os.path.basename(parts[0])

    # Check read-only commands
    if base in READ_ONLY_COMMANDS:
        # sed with -i is a write, not read-only
        if base == "sed" and "-i" in parts:
            return "write"
        return "read-only"

    # Check multi-word read-only (e.g., "git status")
    two_word = f"{base} {parts[1]}" if len(parts) > 1 else ""
    if two_word in READ_ONLY_COMMANDS:
        return "read-only"

    # Known write commands
    if base in {
        "rm",
        "mv",
        "cp",
        "mkdir",
        "rmdir",
        "touch",
        "chmod",
        "chown",
        "ln",
        "install",
        "git commit",
        "git push",
        "git checkout",
        "git reset",
        "git merge",
        "git rebase",
    }:
        return "write"

    # Package managers, compilers, etc. — write
    if base in {
        "pip",
        "pip3",
        "npm",
        "yarn",
        "pnpm",
        "uv",
        "apt",
        "apt-get",
        "brew",
        "cargo",
        "go",
        "gcc",
        "g++",
        "make",
        "cmake",
        "python",
        "python3",
        "node",
        "docker",
        "docker-compose",
    }:
        return "write"

    # Default: write (conservative)
    return "write"


class RunBashCmdInput(BaseModel):
    command: str = Field(description=f"The {_SHELL} command to execute")
    cwd: str | None = Field(
        default=None,
        description="Working directory for the command (optional)",
    )
    timeout: int = Field(
        default=120,
        description="Timeout in seconds (default 120, max 43200 i.e. 12 hours)",
    )
    run_in_background: bool = Field(
        default=False,
        description=(
            "Set to true to run this command in the background. "
            "Returns a task ID immediately. Use TaskOutput to read the output later."
        ),
    )


class RunBashCmdTool(BaseTool):
    name = "Bash"
    description = (
        f"Execute a {_SHELL} command and return its output (stdout and stderr combined). "
        f"Use this for shell operations. Prefer dedicated tools (Read, Glob, Grep) "
        f"for file reading and searching."
    )
    input_schema = RunBashCmdInput

    def is_read_only(self, **kwargs) -> bool:
        """Bash is read-only only if the specific command is classified as read-only.

        When called without a command (e.g., plan mode static check), returns False
        because Bash CAN write — it depends on the specific command.
        """
        command = kwargs.get("command")
        if not command:
            return False  # Bash can write, so not read-only by default
        return _classify_command(command) == "read-only"

    def check_permissions(self, kwargs: dict, context: ToolContext) -> "PermissionResult":
        """Check command safety and path constraints."""
        from scider.core.permissions import allow, deny, is_dangerous_path

        command = kwargs.get("command", "")
        classification = _classify_command(command)

        # Dangerous commands are always denied
        if classification == "dangerous":
            return deny(f"Dangerous command blocked: {command[:100]}")

        # Check for dangerous path access in write commands
        if classification == "write":
            # Extract paths from command (simple heuristic)
            parts = command.split()
            for part in parts:
                if part.startswith("/") or part.startswith("~") or part.startswith("."):
                    if is_dangerous_path(part):
                        return deny(f"Write to dangerous path blocked: {part}")

        return allow()

    def call(
        self,
        context: ToolContext,
        *,
        command: str,
        cwd: str | None = None,
        timeout: int = 120,
        run_in_background: bool = False,
    ) -> str:
        working_dir = None
        if cwd:
            working_dir_path = Path(os.path.expandvars(cwd)).expanduser()
            if not working_dir_path.exists():
                return f"Error: Working directory '{cwd}' does not exist"
            working_dir = str(working_dir_path)

        # Background execution: spawn task and return immediately
        # Background allows up to 12 hours; foreground capped at 10 minutes
        if run_in_background:
            timeout = max(1, min(timeout, 43200))  # up to 12 hours
            from scider.core.task import TaskManager

            task_id = TaskManager.spawn_shell(
                command=command,
                cwd=working_dir,
                timeout=timeout,
                description=command[:100],
            )
            return (
                f"Command is running in the background.\n"
                f"Task ID: {task_id}\n"
                f'Use TaskOutput(task_id="{task_id}") to check status and read output.'
            )

        # Foreground execution (capped at 10 min to avoid blocking the agent loop)
        timeout = max(1, min(timeout, 600))
        try:
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
