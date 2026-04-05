"""Plan mode tools."""

from ..registry import register_new_tool
from .enter_plan_mode import EnterPlanModeTool
from .exit_plan_mode import ExitPlanModeTool

register_new_tool(EnterPlanModeTool())
register_new_tool(ExitPlanModeTool())
