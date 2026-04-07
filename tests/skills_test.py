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
            "when_to_use: When testing\n"
            "allowed_agents: [data, experiment]\n"
            "preload_for: [experiment]\n"
            "---\n\n"
            "Skill content here.\n"
        )
        skill = _parse_skill_file(skill_file)
        assert skill is not None
        assert skill.name == "test-skill"
        assert skill.description == "A test skill"
        assert skill.when_to_use == "When testing"
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
