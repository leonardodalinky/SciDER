"""Skills system — discover, load, and inject skill prompts.

Skills are markdown files with YAML frontmatter that provide domain-specific
guidance to agents. Modeled after Claude Code's skills system (Chapter 9).

Two injection modes:
- **Preloaded**: Full skill content injected into system prompt for agents
  listed in `preload_for`. Always available, no extra tool call needed.
- **On-demand**: Only skill name + description in system prompt. Agent calls
  SkillTool to load full content when needed.

Skill sources (priority order):
1. Project-level: .scider/skills/ (in working directory)
2. User-level: ~/.scider/skills/ (global)

Supported formats:
- skill-name/SKILL.md  (directory format, allows resource files)
- skill-name.md        (simple file format)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import yaml
from loguru import logger


@dataclass
class Skill:
    """A loaded skill."""

    name: str
    description: str
    when_to_use: str
    content: str  # full markdown content (after frontmatter)
    allowed_agents: list[str] | None  # None = all agents
    preload_for: list[str] | None  # Agents that get full content in system prompt
    source_path: str
    source: str = "project"  # "project" or "user"


def _parse_skill_file(path: Path, source: str = "project") -> Skill | None:
    """Parse a skill markdown file with YAML frontmatter."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to read skill file {}: {}", path, e)
        return None

    # Split frontmatter and content
    frontmatter = {}
    content = text
    if text.startswith("---"):
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if match:
            try:
                frontmatter = yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError as e:
                logger.warning("Invalid YAML frontmatter in {}: {}", path, e)
            content = match.group(2)

    # Name: from frontmatter, or from filename/dirname
    name = frontmatter.get("name")
    if not name:
        name = path.parent.name if path.name == "SKILL.md" else path.stem

    # Parse list fields
    def _parse_list(val):
        if val is None:
            return None
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            return [a.strip() for a in val.split(",")]
        return None

    return Skill(
        name=name,
        description=frontmatter.get("description", ""),
        when_to_use=frontmatter.get("when_to_use", ""),
        content=content.strip(),
        allowed_agents=_parse_list(frontmatter.get("allowed_agents")),
        preload_for=_parse_list(frontmatter.get("preload_for")),
        source_path=str(path),
        source=source,
    )


class SkillRegistry:
    """Singleton registry for discovered skills."""

    _instance: SkillRegistry | None = None
    _lock: RLock = RLock()

    def __new__(cls) -> SkillRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._skills: dict[str, Skill] = {}
        self._loaded_dirs: set[str] = set()

    @classmethod
    def instance(cls) -> SkillRegistry:
        return cls()

    def load_from_directory(self, path: str | Path, source: str = "project") -> int:
        """Scan a directory for skills. Returns number loaded. Idempotent."""
        path = Path(path)
        if not path.is_dir():
            return 0

        real_path = str(path.resolve())
        if real_path in self._loaded_dirs:
            return 0
        self._loaded_dirs.add(real_path)

        count = 0
        for entry in sorted(path.iterdir()):
            skill = None
            if entry.is_dir():
                skill_file = entry / "SKILL.md"
                if skill_file.is_file():
                    skill = _parse_skill_file(skill_file, source=source)
            elif entry.is_file() and entry.suffix == ".md":
                skill = _parse_skill_file(entry, source=source)

            if skill:
                if skill.name in self._skills:
                    existing = self._skills[skill.name]
                    if not (source == "project" and existing.source == "user"):
                        continue
                self._skills[skill.name] = skill
                count += 1
                logger.debug("Loaded skill '{}' from {}", skill.name, skill.source_path)

        if count:
            logger.info("Loaded {} skills from {}", count, path)
        return count

    def load_default_directories(self) -> int:
        """Load skills from default directories (project + user level)."""
        total = 0
        total += self.load_from_directory(".scider/skills", source="project")
        user_dir = Path.home() / ".scider" / "skills"
        total += self.load_from_directory(user_dir, source="user")
        return total

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_skills_for_agent(self, agent_name: str) -> list[Skill]:
        """Get all skills available to a specific agent."""
        return [
            s
            for s in self._skills.values()
            if s.allowed_agents is None or agent_name in s.allowed_agents
        ]

    def get_preloaded_skills(self, agent_name: str) -> list[Skill]:
        """Get skills that should be fully injected into system prompt for this agent."""
        return [
            s
            for s in self.get_skills_for_agent(agent_name)
            if s.preload_for is not None and agent_name in s.preload_for
        ]

    def get_ondemand_skills(self, agent_name: str) -> list[Skill]:
        """Get skills available on-demand (not preloaded) for this agent."""
        return [
            s
            for s in self.get_skills_for_agent(agent_name)
            if s.preload_for is None or agent_name not in s.preload_for
        ]

    def build_system_prompt_section(self, agent_name: str) -> str:
        """Build the complete skills section for the system prompt.

        - Preloaded skills: full content included
        - On-demand skills: name + description listing only (use Skill tool to load)
        """
        preloaded = self.get_preloaded_skills(agent_name)
        ondemand = self.get_ondemand_skills(agent_name)

        if not preloaded and not ondemand:
            return ""

        parts = []

        # Preloaded: full content
        if preloaded:
            for skill in preloaded:
                header = f"## Skill: {skill.name}"
                if skill.description:
                    header += f"\n{skill.description}"
                parts.append(f"{header}\n\n{skill.content}")

        # On-demand: listing only
        if ondemand:
            listing_lines = ["## Available Skills (use Skill tool to load)"]
            for skill in ondemand:
                desc = skill.description
                if skill.when_to_use:
                    desc += f" — {skill.when_to_use}"
                listing_lines.append(f"- **{skill.name}**: {desc}")
            parts.append("\n".join(listing_lines))

        return "\n\n---\n\n".join(parts)

    def get_skill_prompts(self, agent_name: str) -> str:
        """Backward-compatible: return full content for all skills (used by Claude SDK)."""
        skills = self.get_skills_for_agent(agent_name)
        if not skills:
            return ""
        parts = []
        for skill in skills:
            header = f"## Skill: {skill.name}"
            if skill.description:
                header += f"\n{skill.description}"
            parts.append(f"{header}\n\n{skill.content}")
        return "\n\n---\n\n".join(parts)

    def list_skills(self) -> list[dict]:
        """List all loaded skills as dicts."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "allowed_agents": s.allowed_agents,
                "preload_for": s.preload_for,
                "source": s.source,
            }
            for s in self._skills.values()
        ]

    def clear(self) -> None:
        """Clear all loaded skills (for testing)."""
        self._skills.clear()
        self._loaded_dirs.clear()
