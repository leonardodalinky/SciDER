"""TaskStop tool — stop a running background task.

Modeled after Claude Code's TaskStopTool.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..base import BaseTool, ToolContext


class TaskStopInput(BaseModel):
    task_id: str = Field(description="The task ID to stop")


class TaskStopTool(BaseTool):
    name = "TaskStop"
    description = (
        "Stop a running background task. "
        "Use when a background command is taking too long or is no longer needed."
    )
    input_schema = TaskStopInput

    def call(self, context: ToolContext, *, task_id: str) -> str:
        from scider.core.task import TaskManager

        return TaskManager.stop(task_id)
