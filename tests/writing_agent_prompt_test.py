"""Tests that the writing_agent prompt YAML renders cleanly and the
SciDER skill registry binds the six PaperOrchestra skills exclusively to
the writing role.
"""

from __future__ import annotations

import pytest

from scider.core.skills import SkillRegistry
from scider.prompts import PROMPTS


class TestWritingAgentPromptRender:
    def test_renders_with_vision_enabled(self):
        rendered = PROMPTS.writing_agent.system_prompt.render(supports_vision=True)
        assert "PaperOrchestra" in rendered
        # Vision block present
        assert "Visual verification (you have image input)" in rendered
        # Step-by-step workflow landmarks
        assert "outline-agent" in rendered
        assert "literature-review-agent" in rendered
        assert "plotting-agent" in rendered
        assert "section-writing-agent" in rendered
        assert "content-refinement-agent" in rendered
        assert "latexmk" in rendered

    def test_renders_without_vision(self):
        rendered = PROMPTS.writing_agent.system_prompt.render(supports_vision=False)
        assert "PaperOrchestra" in rendered
        assert "Visual verification (you have image input)" not in rendered

    def test_default_render_omits_vision_block(self):
        rendered = PROMPTS.writing_agent.system_prompt.render()
        assert "Visual verification (you have image input)" not in rendered

    def test_summary_prompts_exist_and_render(self):
        system = PROMPTS.writing_agent.summary_system_prompt.render()
        user = PROMPTS.writing_agent.summary_user_prompt.render()
        assert "Deliverable" in system
        assert "Pipeline status" in system
        assert user  # non-empty


class TestPaperSkillBindingsForWriting:
    """Verify that the six PaperOrchestra skills show up for writing agent
    and do NOT leak into other agents' prompt-level skill listings.
    """

    @pytest.fixture
    def registry(self) -> SkillRegistry:
        reg = SkillRegistry.instance()
        reg.clear()
        reg.load_default_directories("/home/link/github/SciEvo")
        return reg

    _PAPER_SKILLS = {
        "paper-orchestra",
        "outline-agent",
        "plotting-agent",
        "literature-review-agent",
        "section-writing-agent",
        "content-refinement-agent",
    }

    def test_writing_agent_sees_all_six_paper_skills(self, registry: SkillRegistry):
        writing_skills = {s.name for s in registry.get_skills_for_agent("writing")}
        missing = self._PAPER_SKILLS - writing_skills
        assert not missing, f"writing agent missing paper skills: {missing}"

    def test_paper_orchestra_is_preloaded_for_writing(self, registry: SkillRegistry):
        preloaded = {s.name for s in registry.get_preloaded_skills("writing")}
        assert "paper-orchestra" in preloaded

    def test_sub_skills_are_ondemand_for_writing(self, registry: SkillRegistry):
        ondemand = {s.name for s in registry.get_ondemand_skills("writing")}
        for name in self._PAPER_SKILLS - {"paper-orchestra"}:
            assert name in ondemand, f"{name} should be on-demand for writing, not preloaded"

    def test_paper_skills_not_leaked_to_other_agents(self, registry: SkillRegistry):
        for other_agent in ("data", "experiment", "ideation", "experiment_coding", "critic"):
            other_skills = {s.name for s in registry.get_skills_for_agent(other_agent)}
            leak = other_skills & self._PAPER_SKILLS
            assert not leak, (
                f"paper skills leaked to {other_agent}: {leak} — check "
                f"`allowed_agents` in each SKILL.md frontmatter"
            )
