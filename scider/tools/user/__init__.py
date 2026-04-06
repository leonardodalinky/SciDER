"""User interaction tools."""

from ..registry import register_new_tool
from .ask_user import AskUserQuestionTool

register_new_tool(AskUserQuestionTool())
