"""Tests for scider.core.skills — skill loading, filtering, and SkillTool."""

import tempfile
from pathlib import Path

from scider.core.skills import Skill, SkillRegistry, _parse_skill_file
from scider.tools.base import ToolContext
from scider.tools.registry import ToolRegistry


class TestParseSkillFile:
    def test_parse_with_frontmatter(self, tmp_path):
        skill_file = tmp_path / "test.md"
        skill_file.write_text(
            "---\n"
            "name: test-skill\n"
            "description: A test skill\n"
            "allowed_agents: [data, experiment]\n"
            "preload_for: [experiment]\n"
            "---\n\n"
            "Skill content here.\n"
        )
        skill = _parse_skill_file(skill_file)
        assert skill is not None
        assert skill.name == "test-skill"
        assert skill.description == "A test skill"
        assert skill.allowed_agents == ["data", "experiment"]
        assert skill.preload_for == ["experiment"]
        assert "Skill content here." in skill.content

    def test_parse_without_frontmatter(self, tmp_path):
        skill_file = tmp_path / "simple.md"
        skill_file.write_text("Just plain content.\n")
        skill = _parse_skill_file(skill_file)
        assert skill is not None
        assert skill.name == "simple"
        assert skill.description == ""
        assert "Just plain content." in skill.content

    def test_parse_directory_format(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: Dir skill\n---\n\nDir content.\n")
        skill = _parse_skill_file(skill_dir / "SKILL.md")
        assert skill is not None
        assert skill.name == "my-skill"  # from directory name
        assert skill.description == "Dir skill"

    def test_parse_nonexistent(self, tmp_path):
        skill = _parse_skill_file(tmp_path / "nope.md")
        assert skill is None


class TestSkillRegistry:
    def setup_method(self):
        SkillRegistry.instance().clear()

    def test_load_directory(self, tmp_path):
        (tmp_path / "a.md").write_text("---\nname: alpha\ndescription: A\n---\nContent A\n")
        (tmp_path / "b.md").write_text("---\nname: beta\ndescription: B\n---\nContent B\n")

        count = SkillRegistry.instance().load_from_directory(tmp_path)
        assert count == 2

    def test_idempotent_load(self, tmp_path):
        (tmp_path / "a.md").write_text("---\nname: alpha\n---\nA\n")
        SkillRegistry.instance().load_from_directory(tmp_path)
        count2 = SkillRegistry.instance().load_from_directory(tmp_path)
        assert count2 == 0  # already loaded

    def test_agent_filtering(self, tmp_path):
        (tmp_path / "exp-only.md").write_text(
            "---\nname: exp\nallowed_agents: [experiment]\n---\nFor experiment\n"
        )
        (tmp_path / "all.md").write_text("---\nname: all\n---\nFor everyone\n")
        SkillRegistry.instance().load_from_directory(tmp_path)

        exp_skills = SkillRegistry.instance().get_skills_for_agent("experiment")
        data_skills = SkillRegistry.instance().get_skills_for_agent("data")
        assert len(exp_skills) == 2  # exp + all
        assert len(data_skills) == 1  # only all

    def test_preload_vs_ondemand(self, tmp_path):
        (tmp_path / "preloaded.md").write_text(
            "---\nname: pre\npreload_for: [data]\nallowed_agents: [data]\n---\nPreloaded\n"
        )
        (tmp_path / "ondemand.md").write_text(
            "---\nname: lazy\nallowed_agents: [data]\n---\nOn demand\n"
        )
        SkillRegistry.instance().load_from_directory(tmp_path)

        preloaded = SkillRegistry.instance().get_preloaded_skills("data")
        ondemand = SkillRegistry.instance().get_ondemand_skills("data")
        assert len(preloaded) == 1
        assert preloaded[0].name == "pre"
        assert len(ondemand) == 1
        assert ondemand[0].name == "lazy"

    def test_system_prompt_section(self, tmp_path):
        (tmp_path / "pre.md").write_text(
            "---\nname: pre\ndescription: Preloaded\n"
            "preload_for: [data]\nallowed_agents: [data]\n---\nFull content here\n"
        )
        (tmp_path / "lazy.md").write_text(
            "---\nname: lazy\ndescription: On demand skill\n"
            "allowed_agents: [data]\n---\nLazy content\n"
        )
        SkillRegistry.instance().load_from_directory(tmp_path)

        section = SkillRegistry.instance().build_system_prompt_section("data")
        assert "Full content here" in section  # preloaded: full content
        assert "Available Skills" in section  # on-demand: listing
        assert "lazy" in section

    def test_register_skill_dirs_single(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: A skill\n---\nContent\n"
        )

        count = SkillRegistry.instance().register_skill_dirs(
            skill_dir,
            allow=["experiment"],
            preload_for=["experiment"],
        )
        assert count == 1
        skill = SkillRegistry.instance().get_skill("my-skill")
        assert skill is not None
        assert skill.allowed_agents == ["experiment"]
        assert skill.preload_for == ["experiment"]

    def test_register_skill_dirs_list(self, tmp_path):
        for name in ("skill-a", "skill-b"):
            d = tmp_path / name
            d.mkdir()
            (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {name}\n---\nx\n")

        count = SkillRegistry.instance().register_skill_dirs(
            [tmp_path / "skill-a", tmp_path / "skill-b"],
            allow=["data", "ideation"],
        )
        assert count == 2
        a = SkillRegistry.instance().get_skill("skill-a")
        b = SkillRegistry.instance().get_skill("skill-b")
        assert a.allowed_agents == ["data", "ideation"]
        assert a.preload_for is None  # no preload
        assert b.allowed_agents == ["data", "ideation"]

    def test_register_skill_dirs_overrides_frontmatter(self, tmp_path):
        """Non-None args override frontmatter; None args preserve it."""
        skill_dir = tmp_path / "s"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: s\nallowed_agents: [critic]\npreload_for: [critic]\n---\nx\n"
        )

        SkillRegistry.instance().register_skill_dirs(
            skill_dir,
            allow=["experiment"],
            preload_for=None,
        )
        skill = SkillRegistry.instance().get_skill("s")
        assert skill.allowed_agents == ["experiment"]  # non-None → override
        assert skill.preload_for == ["critic"]  # None → keep frontmatter

    def test_register_skill_dirs_all_none_no_frontmatter(self, tmp_path):
        """None args + no frontmatter → available to all agents, on-demand."""
        skill_dir = tmp_path / "bare"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: bare\n---\nx\n")

        SkillRegistry.instance().register_skill_dirs(skill_dir)
        skill = SkillRegistry.instance().get_skill("bare")
        assert skill.allowed_agents is None  # all agents
        assert skill.preload_for is None  # on-demand

    def test_register_skill_dirs_missing_skill_md(self, tmp_path):
        empty = tmp_path / "empty-dir"
        empty.mkdir()
        count = SkillRegistry.instance().register_skill_dirs(empty, allow=["data"])
        assert count == 0


class TestSkillTool:
    def setup_method(self):
        SkillRegistry.instance().clear()

    def test_load_skill(self, tmp_path):
        (tmp_path / "test.md").write_text("---\nname: test\n---\nTest content\n")
        SkillRegistry.instance().load_from_directory(tmp_path)

        wrapper = ToolRegistry.instance().tools["Skill"].func
        ctx = ToolContext(agent_name="data")
        result = wrapper(skill="test", __tool_context__=ctx)
        assert "Test content" in result

    def test_skill_not_found(self):
        wrapper = ToolRegistry.instance().tools["Skill"].func
        ctx = ToolContext(agent_name="data")
        result = wrapper(skill="nonexistent", __tool_context__=ctx)
        assert "not found" in result.lower()

    def test_args_substitution(self, tmp_path):
        (tmp_path / "sub.md").write_text("---\nname: sub\n---\nAnalyze $ARGUMENTS\n")
        SkillRegistry.instance().load_from_directory(tmp_path)

        wrapper = ToolRegistry.instance().tools["Skill"].func
        ctx = ToolContext(agent_name="data")
        result = wrapper(skill="sub", args="the dataset", __tool_context__=ctx)
        assert "Analyze the dataset" in result
