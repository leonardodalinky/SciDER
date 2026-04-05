"""History management toolset."""

from ..registry import register_new_tool
from .recall_history_patch import RecallHistoryTool

register_new_tool(RecallHistoryTool())
