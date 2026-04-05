"""ExecCtrlcTool — send Ctrl-C to interrupt a running command."""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel

from ...base import BaseTool, ToolContext


class ExecCtrlcInput(BaseModel):
    pass


class ExecCtrlcTool(BaseTool):
    name = "exec_ctrlc"
    description = "Send Ctrl-C to the execution session to interrupt the current command."
    input_schema = ExecCtrlcInput

    def call(self, context: ToolContext) -> str:
        try:
            session = context.agent_state.session
            ctx = session.get_current_context()

            if ctx is None:
                return "No command is currently running"
            elif ctx.is_completed():
                return "The current command has already completed"

            ctx.cancel()
            return "Ctrl-C sent successfully"
        except Exception as e:
            logger.error(f"Error sending Ctrl-C: {e}")
            return f"Error sending Ctrl-C: {e}"
