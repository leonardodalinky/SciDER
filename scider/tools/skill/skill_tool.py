"""SkillTool — load a skill's full content on demand.

Modeled after Claude Code's SkillTool. Skills are listed in the system prompt
by name + description. When the agent needs a skill's full guidance, it calls
this tool to load the complete content.

Preloaded skills (with `preload_for` matching the current agent) are already
in the system prompt — calling this tool for them returns the content but is
redundant.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..base import BaseTool, ToolContext


class SkillInput(BaseModel):
    skill: str = Field(
        description="The skill name to load (as listed in the Available Skills section)",
    )
    args: str | None = Field(
        default=None,
        description="Optional arguments to pass to the skill (replaces $ARGUMENTS in content)",
    )


class SkillTool(BaseTool):
    name = "Skill"
    description = (
        "Load a skill's full content. Skills provide specialized domain knowledge "
        "and guidance. Use when you see a skill listed in 'Available Skills' that "
        "matches the current task."
    )
    input_schema = SkillInput
    _always_read_only = True

    prompt = (
        "# Skill tool usage\n"
        "- Available skills are listed in the system prompt under 'Available Skills'.\n"
        '- Call `Skill(skill="skill-name")` to load the full content of a skill.\n'
        "- Only load skills when you actually need their guidance for the current task.\n"
        "- Skills that are preloaded are already in your context — no need to load them again.\n"
    )

    def call(self, context: ToolContext, *, skill: str, args: str | None = None) -> str:
        from scider.core.skills import SkillRegistry

        s = SkillRegistry.instance().get_skill(skill)
        if s is None:
            # Try case-insensitive match
            for name, sk in SkillRegistry.instance()._skills.items():
                if name.lower() == skill.lower():
                    s = sk
                    break

        if s is None:
            available = [
                sk.name for sk in SkillRegistry.instance().get_skills_for_agent(context.agent_name)
            ]
            return (
                f"Skill '{skill}' not found. "
                f"Available skills: {', '.join(available) if available else 'none'}"
            )

        # Check agent access
        if s.allowed_agents is not None and context.agent_name not in s.allowed_agents:
            return f"Skill '{skill}' is not available for the {context.agent_name} agent."

        content = s.content
        # Substitute $ARGUMENTS if provided
        if args:
            content = content.replace("$ARGUMENTS", args)

        return f"## Skill: {s.name}\n\n{content}"
