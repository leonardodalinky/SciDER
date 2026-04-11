"""Tests that agent prompts conditionally include the vision guidance block.

The experiment, data, and native coding subagent prompts all have an
``{% if supports_vision %}`` section that is only rendered when the
bound model supports image input. These tests exercise both sides of
that toggle for each of the three prompts.
"""

from __future__ import annotations

from scider.prompts import PROMPTS

_VISION_MARKERS = {
    "experiment_agent": "Visual results (you have image input)",
    "data_agent": "Visual inspection (you have image input)",
    "coding_subagent_native": "Figures (you have image input)",
}


class TestVisionPromptBlocks:
    def test_experiment_with_vision(self):
        rendered = PROMPTS.experiment_agent.system_prompt.render(
            coding_backend="native", supports_vision=True
        )
        assert _VISION_MARKERS["experiment_agent"] in rendered
        assert "imgs/" in rendered
        assert "matplotlib" in rendered

    def test_experiment_without_vision(self):
        rendered = PROMPTS.experiment_agent.system_prompt.render(
            coding_backend="native", supports_vision=False
        )
        assert _VISION_MARKERS["experiment_agent"] not in rendered

    def test_data_with_vision(self):
        rendered = PROMPTS.data_agent.system_prompt.render(supports_vision=True)
        assert _VISION_MARKERS["data_agent"] in rendered
        assert "Read" in rendered

    def test_data_without_vision(self):
        rendered = PROMPTS.data_agent.system_prompt.render(supports_vision=False)
        assert _VISION_MARKERS["data_agent"] not in rendered

    def test_coding_subagent_with_vision(self):
        rendered = PROMPTS.coding_subagent_native.system_prompt.render(supports_vision=True)
        assert _VISION_MARKERS["coding_subagent_native"] in rendered
        assert "imgs/" in rendered

    def test_coding_subagent_without_vision(self):
        rendered = PROMPTS.coding_subagent_native.system_prompt.render(supports_vision=False)
        assert _VISION_MARKERS["coding_subagent_native"] not in rendered

    def test_default_render_omits_vision_block(self):
        """When callers don't pass ``supports_vision``, Jinja2 treats it as
        falsy, so the vision block must NOT appear. This guards against a
        change that might accidentally enable vision guidance for
        legacy callers that don't know about the new flag.
        """
        # Only the data_agent template has no required vars; the others
        # need coding_backend. Pick whichever prompt has no required args
        # and verify the default behavior.
        rendered = PROMPTS.data_agent.system_prompt.render()
        assert _VISION_MARKERS["data_agent"] not in rendered
