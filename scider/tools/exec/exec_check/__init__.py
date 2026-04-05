"""ExecCheckTool — check the running state of the current command."""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel

from ...base import BaseTool, ToolContext

MAX_COMMAND_OUTPUT_LENGTH = 16000


class ExecCheckInput(BaseModel):
    pass


class ExecCheckTool(BaseTool):
    name = "exec_check"
    description = (
        "Check the running state and output of the current command in the execution session."
    )
    input_schema = ExecCheckInput
    _always_read_only = True

    def call(self, context: ToolContext) -> str:
        try:
            session = context.agent_state.session
            ctx = session.get_current_context()

            if ctx is None:
                return "No command is currently running"

            if ctx.is_running():
                result = ctx.get_input_output(MAX_COMMAND_OUTPUT_LENGTH)
                return f"Command of `{ctx.command}` is still running...\nCurrent input & output:\n{result}"
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
