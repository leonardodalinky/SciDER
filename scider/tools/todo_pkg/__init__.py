"""Todo tools."""

from ..registry import register_new_tool
from .todo import TodoTool

register_new_tool(TodoTool())
