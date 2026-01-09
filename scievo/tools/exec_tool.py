"""
This module provides the function calling for only the `scievo.core.exec` tool. This should be used only for Experiment Execution Agent.
"""

from typing import TYPE_CHECKING

from loguru import logger

from .registry import register_tool, register_toolset_desc

if TYPE_CHECKING:
    from scievo.core.types import ExecState

register_toolset_desc("exec", "Execution session management toolset for command execution.")

MAX_COMMAND_OUTPUT_LENGTH = 16000


@register_tool(
    "exec",
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Execute a command in the given execution session and wait for it to complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute",
                    },
                },
                "required": ["command"],
            },
        },
    },
)
def exec_command(agent_state: "ExecState", command: str) -> str:
    """Execute a command in the given execution session."""
    try:
        ctx = agent_state.session.exec(command, timeout=None)

        TIMEOUT = 3.0

        # Wait for the command to complete, which is used to simplify the fast case
        is_finished = ctx.wait(timeout=TIMEOUT)

        if not is_finished or ctx.is_running():
            result = ctx.get_input_output(max_length=MAX_COMMAND_OUTPUT_LENGTH)
            return f"WARNING: Command execution of `{command}` is not finished in {TIMEOUT} seconds. Try to check the execution status later.\nCurrent input & output:\n---\n{result}"

        # Get the result
        result = ctx.get_input_output(max_length=MAX_COMMAND_OUTPUT_LENGTH)

        # Check for errors
        if ctx.has_error():
            error_msg = ctx.get_error()
            return f"ERROR: Command execution of `{command}`.\nError message: {error_msg}\nCommand input & output:\n---\n{result}"

        return result
    except Exception as e:
        logger.error(f"Error executing command of `{command}`: {e}")
        return f"Error executing command of `{command}`: {e}"


@register_tool(
    "exec",
    {
        "type": "function",
        "function": {
            "name": "exec_ctrlc",
            "description": "Send Ctrl-C to the execution session to interrupt the current command.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
)
def exec_ctrlc(agent_state: "ExecState") -> str:
    """Send Ctrl-C to the execution session."""
    try:
        session = agent_state.session
        ctx = session.get_current_context()

        if ctx is None:
            return "No command is currently running"
        elif ctx.is_completed():
            return "The current command has already completed"

        # Cancel the current command (which sends Ctrl-C)
        ctx.cancel()

        return "Ctrl-C sent successfully"
    except Exception as e:
        logger.error(f"Error sending Ctrl-C: {e}")
        return f"Error sending Ctrl-C: {e}"


@register_tool(
    "exec",
    {
        "type": "function",
        "function": {
            "name": "exec_check",
            "description": "Check the running state and output of the current command in the execution session.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
)
def exec_check(agent_state: "ExecState") -> str:
    """Check the running state of the current command."""
    try:
        session = agent_state.session
        ctx = session.get_current_context()

        if ctx is None:
            return "No command is currently running"

        # Check the state
        if ctx.is_running():
            result = ctx.get_input_output(MAX_COMMAND_OUTPUT_LENGTH)
            return (
                f"Command of `{ctx.command}` is still running...\nCurrent input & output:\n{result}"
            )
        elif ctx.is_completed():
            result = ctx.get_input_output(MAX_COMMAND_OUTPUT_LENGTH)
            return f"Command of `{ctx.command}` completed successfully:\n{result}"
        elif ctx.has_error():
            result = ctx.get_input_output(MAX_COMMAND_OUTPUT_LENGTH)
            error_msg = ctx.get_error()
            return f"Command of `{ctx.command}` failed with error: {error_msg}\n{result}"
        else:
            logger.error(f"Unknown command state of 'exec_check' for `{ctx.command}`")
            return "Unknown command state"
    except Exception as e:
        logger.error(f"Error checking command state: {e}")
        return f"Error checking command state: {e}"
