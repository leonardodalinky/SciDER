"""
Claude Code toolset.

This toolset provides a thin wrapper around the local `claude` CLI (Claude Code)
so agents can delegate codebase edits to it.

Notes:
- This repository does NOT vendor Claude Code. Users must install/configure it.
- The wrapper is best-effort and intentionally generic across CLI versions.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel

from scider.core import constant


class ClaudeCodeResult(BaseModel):
    command: str
    cwd: str
    returncode: int
    stdout: str
    stderr: str


_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text or "")


def _resolve_cwd(cwd: str | None, agent_state) -> Path:
    if cwd:
        p = Path(os.path.expandvars(cwd)).expanduser()
        return p.resolve()
    if agent_state is not None:
        if hasattr(agent_state, "local_env") and hasattr(agent_state.local_env, "working_dir"):
            try:
                return Path(agent_state.local_env.working_dir).resolve()
            except Exception:
                pass
        if hasattr(agent_state, "repo_dir") and agent_state.repo_dir:
            try:
                return Path(agent_state.repo_dir).resolve()
            except Exception:
                pass
    return Path.cwd().resolve()


def _resolve_claude_cmd() -> list[str] | None:
    """
    Resolve the Claude Code CLI command.

    - If env `CLAUDE_CODE_CMD` is set, it can contain a full command string with flags.
      Example: "claude --print"
    - Otherwise, falls back to `claude` from PATH.
    """
    cmd = os.environ.get("CLAUDE_CODE_CMD")
    if cmd and cmd.strip():
        return shlex.split(cmd)
    if shutil.which("claude"):
        return ["claude"]
    return None


def run_claude_code(
    instruction: str,
    cwd: str | None = None,
    timeout: int = 600,
    extra_args: list[str] | None = None,
    **kwargs,
) -> str:
    try:
        if not instruction or not instruction.strip():
            return json.dumps({"error": "instruction must be a non-empty string"})

        agent_state = kwargs.get(constant.__AGENT_STATE_NAME__)
        working_dir = _resolve_cwd(cwd, agent_state)
        if not working_dir.exists():
            return json.dumps({"error": f"Working directory does not exist: {str(working_dir)}"})
        if not working_dir.is_dir():
            return json.dumps(
                {"error": f"Working directory is not a directory: {str(working_dir)}"}
            )

        base_cmd = _resolve_claude_cmd()
        if not base_cmd:
            return json.dumps(
                {
                    "error": "Claude Code CLI not found (expected `claude` in PATH).",
                    "hint": "Install Claude Code and ensure `claude` is available, or set env `CLAUDE_CODE_CMD` to the full command.",
                }
            )

        # Best-effort invocation:
        # - Many CLIs accept prompts via stdin.
        # - Some accept `--message-file` / `--prompt-file` style flags; we don't assume exact flag names.
        # We do both: pass stdin, and also create a temp file and expose its path via env for advanced wrappers.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(instruction)
            f.flush()
            prompt_file = f.name

        cmd_list = list(base_cmd)

        # Add non-interactive flags to auto-accept edits
        # --print: non-interactive mode, skips workspace trust dialog
        # --permission-mode acceptEdits: auto-accept file edits without prompting
        if "--print" not in cmd_list and "-p" not in cmd_list:
            cmd_list.append("--print")
        if "--permission-mode" not in cmd_list:
            cmd_list.extend(["--permission-mode", "acceptEdits"])

        if extra_args:
            cmd_list.extend(extra_args)

        env = os.environ.copy()
        env["SCIDER_CLAUDE_PROMPT_FILE"] = prompt_file

        proc = subprocess.run(
            cmd_list,
            input=instruction,
            capture_output=True,
            text=True,
            cwd=str(working_dir),
            timeout=timeout,
            env=env,
        )

        result = ClaudeCodeResult(
            command=" ".join(cmd_list),
            cwd=str(working_dir),
            returncode=proc.returncode,
            stdout=_strip_ansi(proc.stdout or ""),
            stderr=_strip_ansi(proc.stderr or ""),
        )

        # Clean up temp prompt file
        try:
            os.unlink(prompt_file)
        except Exception:
            pass

        # Return structured text for LLM consumption
        return json.dumps(result.model_dump())
    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "error": f"Claude Code command timed out after {timeout} seconds",
                "hint": "Try increasing timeout, or provide `CLAUDE_CODE_CMD` with non-interactive flags (if supported).",
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})
