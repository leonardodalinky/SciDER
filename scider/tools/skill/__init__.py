"""Skill tool — load skill content on demand."""

from ..registry import register_new_tool
from .skill_tool import SkillTool

register_new_tool(SkillTool())
