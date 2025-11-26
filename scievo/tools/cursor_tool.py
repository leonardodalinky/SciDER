import subprocess
from pathlib import Path

from pydantic import BaseModel

from ..core import constant
from ..core.utils import wrap_dict_to_toon
from .registry import register_tool, register_toolset_desc

register_toolset_desc(
    "cursor",
    "Cursor CLI control toolset. Note: Requires Cursor editor to be installed and configured with API keys. "
    "API keys are configured in Cursor editor settings (not via CLI).",
)


class CursorResult(BaseModel):
    command: str
    returncode: int
    stdout: str
    stderr: str


def run_shell(cmd: str, cwd: Path | None = None, agent_state=None) -> CursorResult:
    """
    Run a shell command, using the working directory from agent_state if available.

    Args:
        cmd: Command to run
        cwd: Optional working directory (overrides agent_state if provided)
        agent_state: Optional agent state to get working directory from
    """
    # Determine working directory: cwd > agent_state.local_env.working_dir > current directory
    if cwd is None and agent_state is not None:
        if hasattr(agent_state, "local_env") and hasattr(agent_state.local_env, "working_dir"):
            cwd = agent_state.local_env.working_dir
        elif hasattr(agent_state, "repo_dir") and agent_state.repo_dir:
            cwd = Path(agent_state.repo_dir)

    if cwd is None:
        cwd = Path.cwd()

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate()
    return CursorResult(
        command=cmd,
        returncode=proc.returncode,
        stdout=out,
        stderr=err,
    )


@register_tool(
    "cursor",
    {
        "type": "function",
        "function": {
            "name": "cursor_chat",
            "description": "Chat with Cursor CLI to get advice or refactoring suggestions (no direct code edit).",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message sent to Cursor chat.",
                    }
                },
                "required": ["message"],
            },
        },
    },
)
def cursor_chat_tool(message: str, **kwargs) -> str:
    try:
        # Get agent_state from kwargs if available
        agent_state = kwargs.get(constant.__AGENT_STATE_NAME__)
        cmd = f"cursor chat --message {message!r}"
        result = run_shell(cmd, agent_state=agent_state)

        # Check if cursor command failed - might be due to missing installation or API key
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            if "command not found" in error_msg.lower() or "not found" in error_msg.lower():
                return wrap_dict_to_toon(
                    {
                        "error": "Cursor CLI not found. Please ensure Cursor editor is installed and 'cursor' command is available in PATH.",
                        "hint": "Install Cursor from https://cursor.sh and ensure it's added to your PATH.",
                    }
                )
            elif (
                "api" in error_msg.lower()
                or "key" in error_msg.lower()
                or "auth" in error_msg.lower()
            ):
                return wrap_dict_to_toon(
                    {
                        "error": "Cursor API key not configured or invalid.",
                        "hint": "Please configure API keys in Cursor editor settings (Settings > AI > API Keys). "
                        "You can use OpenAI, Anthropic, or other supported providers.",
                    }
                )
            else:
                return wrap_dict_to_toon(result.dict())

        return wrap_dict_to_toon(result.dict())
    except Exception as e:
        return wrap_dict_to_toon({"error": str(e)})


@register_tool(
    "cursor",
    {
        "type": "function",
        "function": {
            "name": "cursor_edit",
            "description": "Ask Cursor CLI to edit files based on a prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Instruction for Cursor to perform code edits.",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of files to limit editing scope.",
                    },
                },
                "required": ["message"],
            },
        },
    },
)
def cursor_edit_tool(message: str, files: list[str] | None = None, **kwargs) -> str:
    try:
        # Get agent_state from kwargs if available
        agent_state = kwargs.get(constant.__AGENT_STATE_NAME__)
        files_arg = " ".join(files) if files else ""
        cmd = f"cursor edit {files_arg} --message {message!r}"

        result = run_shell(cmd, agent_state=agent_state)

        # Check if cursor command failed - might be due to missing installation or API key
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            if "command not found" in error_msg.lower() or "not found" in error_msg.lower():
                return wrap_dict_to_toon(
                    {
                        "error": "Cursor CLI not found. Please ensure Cursor editor is installed and 'cursor' command is available in PATH.",
                        "hint": "Install Cursor from https://cursor.sh and ensure it's added to your PATH.",
                    }
                )
            elif (
                "api" in error_msg.lower()
                or "key" in error_msg.lower()
                or "auth" in error_msg.lower()
            ):
                return wrap_dict_to_toon(
                    {
                        "error": "Cursor API key not configured or invalid.",
                        "hint": "Please configure API keys in Cursor editor settings (Settings > AI > API Keys). "
                        "You can use OpenAI, Anthropic, or other supported providers.",
                    }
                )
            else:
                return wrap_dict_to_toon(result.dict())

        return wrap_dict_to_toon(result.dict())
    except Exception as e:
        return wrap_dict_to_toon({"error": str(e)})


@register_tool(
    "cursor",
    {
        "type": "function",
        "function": {
            "name": "cursor_fix_tests",
            "description": "Let Cursor CLI attempt to fix failing tests.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
)
def cursor_fix_tests_tool(**kwargs) -> str:
    try:
        # Get agent_state from kwargs if available
        agent_state = kwargs.get(constant.__AGENT_STATE_NAME__)
        cmd = "cursor fix-tests"
        result = run_shell(cmd, agent_state=agent_state)

        # Check if cursor command failed - might be due to missing installation or API key
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            if "command not found" in error_msg.lower() or "not found" in error_msg.lower():
                return wrap_dict_to_toon(
                    {
                        "error": "Cursor CLI not found. Please ensure Cursor editor is installed and 'cursor' command is available in PATH.",
                        "hint": "Install Cursor from https://cursor.sh and ensure it's added to your PATH.",
                    }
                )
            elif (
                "api" in error_msg.lower()
                or "key" in error_msg.lower()
                or "auth" in error_msg.lower()
            ):
                return wrap_dict_to_toon(
                    {
                        "error": "Cursor API key not configured or invalid.",
                        "hint": "Please configure API keys in Cursor editor settings (Settings > AI > API Keys). "
                        "You can use OpenAI, Anthropic, or other supported providers.",
                    }
                )
            else:
                return wrap_dict_to_toon(result.dict())

        return wrap_dict_to_toon(result.dict())
    except Exception as e:
        return wrap_dict_to_toon({"error": str(e)})
