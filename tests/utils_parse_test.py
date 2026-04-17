"""Tests for LLM-response parsing helpers in scider.core.utils.

Regression targets — before the fix, the helpers fed the entire LLM message
(prose critique + final fenced JSON) into ``json_repair`` with a broken regex
that matched the whole blob; json_repair then produced a list instead of the
intended dict. These tests pin down the extract-last-fenced-block behavior.
"""

import pytest
from pydantic import BaseModel

from scider.core.utils import (
    parse_json_from_llm_response,
    parse_json_from_text,
    parse_markdown_from_llm_response,
)


class _Spec(BaseModel):
    name: str
    value: int


class TestParseJsonFromText:
    def test_prose_then_fenced_json_picks_json(self):
        """Regression: prose prefix + fenced JSON must ignore the prose."""
        text = (
            "Looking at the data, I find issues.\n\n"
            '```json\n{"verdict": "reject", "feedback": "bad output"}\n```'
        )
        assert parse_json_from_text(text) == {
            "verdict": "reject",
            "feedback": "bad output",
        }

    def test_multiple_fenced_blocks_last_wins(self):
        """LLMs sometimes draft then refine — honor the final block."""
        text = (
            'First draft:\n```json\n{"draft": true}\n```\n'
            'Actually, final:\n```json\n{"final": true}\n```'
        )
        assert parse_json_from_text(text) == {"final": True}

    def test_raw_json_no_fence(self):
        assert parse_json_from_text('{"x": 1}') == {"x": 1}

    def test_pydantic_target(self):
        spec = parse_json_from_text('```json\n{"name": "a", "value": 3}\n```', tgt_type=_Spec)
        assert isinstance(spec, _Spec) and spec.name == "a" and spec.value == 3

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            parse_json_from_text("")

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            parse_json_from_text("no json here at all, just prose")


class TestParseJsonFromLLMResponse:
    def test_prose_then_fenced_json_picks_json(self):
        text = (
            "Sure, here's the spec you asked for:\n"
            '```json\n{"name": "b", "value": 7}\n```\n'
            "Let me know if that works."
        )
        spec = parse_json_from_llm_response(text, tgt_type=_Spec)
        assert spec.name == "b" and spec.value == 7

    def test_raw_json_works(self):
        spec = parse_json_from_llm_response('{"name": "c", "value": 1}', tgt_type=_Spec)
        assert spec.name == "c" and spec.value == 1

    def test_no_tgt_type_returns_raw_dict(self):
        """Without tgt_type, return the parsed dict/list directly."""
        out = parse_json_from_llm_response('```json\n{"k": 1}\n```')
        assert out == {"k": 1}

    def test_accepts_message_input(self):
        """Should unwrap Message.content transparently."""
        from scider.core.types import Message

        msg = Message(role="assistant", content='{"k": 2}')
        assert parse_json_from_llm_response(msg) == {"k": 2}


class TestParseMarkdownFromLLMResponse:
    def test_fenced_markdown_stripped(self):
        text = "Here is my summary:\n" "```markdown\n# Heading\n\nBody paragraph.\n```"
        assert parse_markdown_from_llm_response(text) == "# Heading\n\nBody paragraph."

    def test_no_fence_returns_stripped_text(self):
        assert parse_markdown_from_llm_response("Plain markdown # Heading\n") == (
            "Plain markdown # Heading"
        )

    def test_multiple_fences_last_wins(self):
        text = "Draft:\n```markdown\n# draft\n```\n\nFinal:\n```markdown\n# final\n```"
        assert parse_markdown_from_llm_response(text) == "# final"
