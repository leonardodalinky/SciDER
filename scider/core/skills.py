"""Skills system — discover, load, and inject skill prompts.

Skills are markdown files with YAML frontmatter that provide domain-specific
guidance to agents. Modeled after Claude Code's skills system (Chapter 9).

Two injection modes:
- **Preloaded**: Full skill content injected into system prompt for agents
  listed in `preload_for`. Always available, no extra tool call needed.
- **On-demand**: Only skill name + description in system prompt. Agent calls
  SkillTool to load full content when needed.

Skill sources (priority order, walk-up from workspace to root + home):
  Each .scider/skills/ directory is scanned.

Supported formats:
- .scider/skills/skill-name/SKILL.md  (directory format, allows resource files)
- .scider/skills/skill-name.md        (simple file format)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal, Sequence, Union

import yaml
from loguru import logger

# Canonical agent names that can be targeted by skills.
AgentName = Literal[
    "ideation",
    "data",
    "experiment",
    "experiment_coding",
    "native_coding",
    "critic",
    "paper_search",
]


@dataclass
class Skill:
    """A loaded skill."""

    name: str
    description: str
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

    body = content.strip()

    # For directory-format skills (name/SKILL.md), inject the base directory
    # so the model can resolve relative file references via the Read tool.
    if path.name == "SKILL.md":
        base_dir = str(path.parent.resolve())
        body = f"Base directory for this skill: {base_dir}\n\n{body}"

    return Skill(
        name=name,
        description=frontmatter.get("description", ""),
        content=body,
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

    def load_default_directories(self, workspace_path: str | Path | None = None) -> int:
        """Load skills by walking up from workspace to root, plus home dir.

        At each directory, scans .scider/skills/ for skill subdirs
        (name/SKILL.md) and loose .md files. Directories closer to the
        workspace load later, so project-level skills override
        identically-named ones from parents.
        """
        from .scider_context import walk_up_dirs

        dirs = walk_up_dirs(workspace_path)
        total = 0
        for d in dirs:
            skills_dir = d / ".scider" / "skills"
            # Dirs closer to workspace load later → "project" source wins
            source = "project" if d == dirs[-1] else "user"
            total += self.load_from_directory(skills_dir, source=source)
        return total

    def register_skill_dirs(
        self,
        path: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        allow: Sequence[AgentName] | None = None,
        preload_for: Sequence[AgentName] | None = None,
        source: str = "project",
    ) -> int:
        """Register one or more skill directories with programmatic overrides.

        Each ``path`` must be a directory containing a ``SKILL.md`` file
        (directory-format skill). ``allow`` and ``preload_for`` override
        the matching fields in the skill's frontmatter when provided.

        Args:
            path: A single directory or a list/sequence of directories.
            allow: Agent names allowed to use these skills. If ``None``,
                the frontmatter's ``allowed_agents`` is kept; if the
                frontmatter also has no value, the skill is available to
                all agents.
            preload_for: Agents that get the full skill content in their
                system prompt. If ``None``, the frontmatter's ``preload_for``
                is kept; if the frontmatter also has no value, the skill
                is on-demand only.
            source: "project" (default) or "user", used for override priority.

        Returns:
            Number of skills successfully registered.
        """
        if isinstance(path, (str, Path)):
            paths: list[Path] = [Path(path)]
        else:
            paths = [Path(p) for p in path]

        allow_override = list(allow) if allow is not None else None
        preload_override = list(preload_for) if preload_for is not None else None

        count = 0
        for p in paths:
            skill_file = p / "SKILL.md"
            if not skill_file.is_file():
                logger.warning("No SKILL.md found in {}", p)
                continue

            skill = _parse_skill_file(skill_file, source=source)
            if skill is None:
                continue

            # Programmatic values override frontmatter ONLY when explicitly
            # provided (non-None). None means "keep frontmatter value".
            if allow_override is not None:
                skill.allowed_agents = allow_override
            if preload_override is not None:
                skill.preload_for = preload_override

            if skill.name in self._skills:
                existing = self._skills[skill.name]
                if not (source == "project" and existing.source == "user"):
                    continue
            self._skills[skill.name] = skill
            count += 1
            logger.debug("Registered skill '{}' from {}", skill.name, skill.source_path)

        if count:
            logger.info("Registered {} skills via register_skill_dirs", count)
        return count

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
                listing_lines.append(f"- **{skill.name}**: {skill.description}")
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
