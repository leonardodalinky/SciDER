import json
import os
import subprocess
from pathlib import Path

from ..core import constant
from .registry import register_tool, register_toolset_desc

register_toolset_desc(
    "cursor",
    "Cursor Agent toolset. Uses 'cursor-agent' command to interact with Cursor for code editing, chat, and test fixing. "
    "Note: Requires Cursor editor to be installed and configured with API keys. "
    "API keys are configured in Cursor editor settings (Settings > AI > API Keys).",
)


def _get_working_dir(cwd: Path | None = None, agent_state=None) -> Path:
    """Get working directory from cwd, agent_state, or current directory."""
    if cwd is not None:
        return Path(cwd)

    if agent_state is not None:
        if hasattr(agent_state, "local_env") and hasattr(agent_state.local_env, "working_dir"):
            return Path(agent_state.local_env.working_dir)
        elif hasattr(agent_state, "repo_dir") and agent_state.repo_dir:
            return Path(agent_state.repo_dir)

    return Path.cwd()


def _run_cursor_agent(
    prompt: str,
    cwd: Path | None = None,
    agent_state=None,
    model: str | None = None,
    output_format: str = "json",
    timeout: int = 300,
) -> dict:
    """
    Run cursor-agent command in non-interactive mode.

    Args:
        prompt: The prompt/instruction to send to cursor-agent
        cwd: Optional working directory
        agent_state: Optional agent state to get working directory from
        model: Optional model to use (e.g., "gpt-5", "sonnet-4")
        output_format: Output format ("text", "json", "stream-json"), default "json"
        timeout: Timeout in seconds, default 300 (5 minutes)

    Returns:
        Dictionary with command, returncode, stdout, stderr
    """
    working_dir = _get_working_dir(cwd, agent_state)

    # Build command: cursor-agent -p "prompt" --output-format json
    cmd = ["cursor-agent", "-p", prompt, "--output-format", output_format]

    # Add model if specified
    if model:
        cmd.extend(["--model", model])

    # Add workspace directory
    cmd.extend(["--workspace", str(working_dir)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(working_dir),
            timeout=timeout,
        )

        return {
            "command": " ".join(cmd),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "working_dir": str(working_dir),
        }
    except subprocess.TimeoutExpired:
        return {
            "command": " ".join(cmd),
            "returncode": -1,
            "stdout": "",
            "stderr": f"Error: Command timed out after {timeout} seconds",
            "working_dir": str(working_dir),
        }
    except FileNotFoundError:
        return {
            "command": " ".join(cmd),
            "returncode": -1,
            "stdout": "",
            "stderr": "Error: cursor-agent command not found. Please ensure Cursor editor is installed and 'cursor-agent' command is available in PATH.",
            "working_dir": str(working_dir),
        }
    except Exception as e:
        return {
            "command": " ".join(cmd),
            "returncode": -1,
            "stdout": "",
            "stderr": f"Error executing cursor-agent: {e}",
            "working_dir": str(working_dir),
        }


@register_tool(
    "cursor",
    {
        "type": "function",
        "function": {
            "name": "cursor_agent_edit",
            "description": "Ask Cursor Agent to edit files based on a prompt. The agent can read, modify, and create files in the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Instruction for Cursor agent to perform code edits. Be specific about what files to modify and what changes to make.",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of files to focus on. You can mention these in the message instead.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model to use (e.g., 'gpt-5', 'sonnet-4'). If not provided, uses default model.",
                    },
                },
                "required": ["message"],
            },
        },
    },
)
def cursor_agent_edit(
    message: str, files: list[str] | None = None, model: str | None = None, **kwargs
) -> str:
    """Ask Cursor Agent to edit files based on a prompt."""
    try:
        agent_state = kwargs.get(constant.__AGENT_STATE_NAME__)

        # Build the prompt - include file information if provided
        if files:
            files_info = f"Focus on these files: {', '.join(files)}. "
            prompt = f"{files_info}{message}"
        else:
            prompt = message

        # Run cursor-agent in non-interactive mode with JSON output
        result = _run_cursor_agent(
            prompt, agent_state=agent_state, model=model, output_format="json"
        )

        # Parse JSON output if available
        if result["returncode"] == 0 and result["stdout"]:
            try:
                import json

                json_result = json.loads(result["stdout"].strip())
                if isinstance(json_result, dict):
                    if json_result.get("is_error", False):
                        return json.dumps(
                            {
                                "error": json_result.get("result", "Unknown error"),
                                "success": False,
                                **result,
                            }
                        )
                    else:
                        return json.dumps(
                            {
                                "success": True,
                                "result": json_result.get("result", ""),
                                "session_id": json_result.get("session_id"),
                                **result,
                            }
                        )
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw output
                pass

        # Check for errors
        if result["returncode"] != 0:
            error_msg = result.get("stderr", "") or result.get("stdout", "") or "Unknown error"
            if "command not found" in error_msg.lower() or "not found" in error_msg.lower():
                return json.dumps(
                    {
                        "error": "cursor-agent command not found. Please ensure Cursor editor is installed and 'cursor-agent' command is available in PATH.",
                        "hint": "Install Cursor from https://cursor.sh and ensure it's added to your PATH.",
                        **result,
                    }
                )
            elif (
                "api" in error_msg.lower()
                or "key" in error_msg.lower()
                or "auth" in error_msg.lower()
            ):
                return json.dumps(
                    {
                        "error": "Cursor API key not configured or invalid.",
                        "hint": "Please configure API keys in Cursor editor settings (Settings > AI > API Keys). "
                        "You can use OpenAI, Anthropic, or other supported providers.",
                        **result,
                    }
                )
            else:
                return json.dumps(result)

        # Return result (may contain raw text if JSON parsing failed)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@register_tool(
    "cursor",
    {
        "type": "function",
        "function": {
            "name": "cursor_agent_chat",
            "description": "Chat with Cursor Agent to get advice or refactoring suggestions. The agent can analyze code and provide recommendations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message sent to Cursor agent for advice or suggestions.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model to use (e.g., 'gpt-5', 'sonnet-4'). If not provided, uses default model.",
                    },
                },
                "required": ["message"],
            },
        },
    },
)
def cursor_agent_chat(message: str, model: str | None = None, **kwargs) -> str:
    """Chat with Cursor Agent to get advice or refactoring suggestions."""
    try:
        agent_state = kwargs.get(constant.__AGENT_STATE_NAME__)

        # Run cursor-agent in non-interactive mode with JSON output
        result = _run_cursor_agent(
            message, agent_state=agent_state, model=model, output_format="json"
        )

        # Parse JSON output if available
        if result["returncode"] == 0 and result["stdout"]:
            try:
                import json

                json_result = json.loads(result["stdout"].strip())
                if isinstance(json_result, dict) and "result" in json_result:
                    return json.dumps(
                        {
                            "success": True,
                            "result": json_result.get("result", ""),
                            "session_id": json_result.get("session_id"),
                            **result,
                        }
                    )
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw output
                pass

        # Check for errors
        if result["returncode"] != 0:
            error_msg = result.get("stderr", "") or result.get("stdout", "") or "Unknown error"
            if "command not found" in error_msg.lower() or "not found" in error_msg.lower():
                return json.dumps(
                    {
                        "error": "cursor-agent command not found. Please ensure Cursor editor is installed and 'cursor-agent' command is available in PATH.",
                        "hint": "Install Cursor from https://cursor.sh and ensure it's added to your PATH.",
                        **result,
                    }
                )
            elif (
                "api" in error_msg.lower()
                or "key" in error_msg.lower()
                or "auth" in error_msg.lower()
            ):
                return json.dumps(
                    {
                        "error": "Cursor API key not configured or invalid.",
                        "hint": "Please configure API keys in Cursor editor settings (Settings > AI > API Keys). "
                        "You can use OpenAI, Anthropic, or other supported providers.",
                        **result,
                    }
                )
            else:
                return json.dumps(result)

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@register_tool(
    "cursor",
    {
        "type": "function",
        "function": {
            "name": "cursor_agent_fix_tests",
            "description": "Ask Cursor Agent to fix failing tests. The agent will analyze test failures and attempt to fix them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_command": {
                        "type": "string",
                        "description": "Optional command to run tests (e.g., 'pytest', 'python -m unittest'). If not provided, agent will try to detect.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model to use (e.g., 'gpt-5', 'sonnet-4'). If not provided, uses default model.",
                    },
                },
                "required": [],
            },
        },
    },
)
def cursor_agent_fix_tests(
    test_command: str | None = None, model: str | None = None, **kwargs
) -> str:
    """Ask Cursor Agent to fix failing tests."""
    try:
        agent_state = kwargs.get(constant.__AGENT_STATE_NAME__)

        # Build prompt for fixing tests
        if test_command:
            prompt = f"Run '{test_command}' to check for failing tests, then fix any failing tests."
        else:
            prompt = "Find and run the test suite, then fix any failing tests."

        # Run cursor-agent in non-interactive mode with JSON output
        result = _run_cursor_agent(
            prompt, agent_state=agent_state, model=model, output_format="json"
        )

        # Parse JSON output if available
        if result["returncode"] == 0 and result["stdout"]:
            try:
                import json

                json_result = json.loads(result["stdout"].strip())
                if isinstance(json_result, dict):
                    if json_result.get("is_error", False):
                        return json.dumps(
                            {
                                "error": json_result.get("result", "Unknown error"),
                                "success": False,
                                **result,
                            }
                        )
                    else:
                        return json.dumps(
                            {
                                "success": True,
                                "result": json_result.get("result", ""),
                                "session_id": json_result.get("session_id"),
                                **result,
                            }
                        )
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw output
                pass

        # Check for errors
        if result["returncode"] != 0:
            error_msg = result.get("stderr", "") or result.get("stdout", "") or "Unknown error"
            if "command not found" in error_msg.lower() or "not found" in error_msg.lower():
                return json.dumps(
                    {
                        "error": "cursor-agent command not found. Please ensure Cursor editor is installed and 'cursor-agent' command is available in PATH.",
                        "hint": "Install Cursor from https://cursor.sh and ensure it's added to your PATH.",
                        **result,
                    }
                )
            elif (
                "api" in error_msg.lower()
                or "key" in error_msg.lower()
                or "auth" in error_msg.lower()
            ):
                return json.dumps(
                    {
                        "error": "Cursor API key not configured or invalid.",
                        "hint": "Please configure API keys in Cursor editor settings (Settings > AI > API Keys). "
                        "You can use OpenAI, Anthropic, or other supported providers.",
                        **result,
                    }
                )
            else:
                return json.dumps(result)

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})
