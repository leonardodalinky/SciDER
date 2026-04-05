"""ExecCommandTool — execute a command in an execution session."""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext

MAX_COMMAND_OUTPUT_LENGTH = 16000


class ExecCommandInput(BaseModel):
    command: str = Field(description="The command to execute")


class ExecCommandTool(BaseTool):
    name = "exec_command"
    description = "Execute a command in the given execution session and wait for it to complete."
    input_schema = ExecCommandInput

    def call(self, context: ToolContext, *, command: str) -> str:
        try:
            agent_state = context.agent_state
            ctx = agent_state.session.exec(command, timeout=None)

            TIMEOUT = 3.0
            is_finished = ctx.wait(timeout=TIMEOUT)

            if not is_finished or ctx.is_running():
                result = ctx.get_input_output(max_length=MAX_COMMAND_OUTPUT_LENGTH)
                return (
                    f"WARNING: Command execution of `{command}` is not finished "
                    f"in {TIMEOUT} seconds. Try to check the execution status later.\n"
                    f"Current input & output:\n---\n{result}"
                )

            result = ctx.get_input_output(max_length=MAX_COMMAND_OUTPUT_LENGTH)

            if ctx.has_error():
                error_msg = ctx.get_error()
                return (
                    f"ERROR: Command execution of `{command}`.\n"
                    f"Error message: {error_msg}\n"
                    f"Command input & output:\n---\n{result}"
                )

            return result
        except Exception as e:
            logger.error(f"Error executing command of `{command}`: {e}")
            return f"Error executing command of `{command}`: {e}"
