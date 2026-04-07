"""Task management tools (TaskOutput, TaskStop)."""

from ..registry import register_new_tool
from .task_output import TaskOutputTool
from .task_stop import TaskStopTool

register_new_tool(TaskOutputTool())
register_new_tool(TaskStopTool())
