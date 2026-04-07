"""Shell tools."""

from ..registry import register_new_tool
from .run_bash_cmd import RunBashCmdTool

register_new_tool(RunBashCmdTool())
