"""Tests for provider-aware message serialization in scider.core.llms.

Covers:
- ``_detect_provider`` for common litellm model ids
- ``_serialize_messages_for_provider`` for Anthropic / Gemini / OpenAI /
  other, including the OpenAI synthetic-user-message workaround
- ``_serialize_messages_for_responses_api`` for the GPT-5 Responses path
"""

from __future__ import annotations

from scider.core.llms import (
    _detect_provider,
    _serialize_messages_for_provider,
    _serialize_messages_for_responses_api,
)
from scider.core.types import Message


def _make_tool_msg_with_image(tool_call_id: str = "call_1") -> Message:
    return Message(
        role="tool",
        tool_call_id=tool_call_id,
        tool_name="Read",
        content="[File: /tmp/img.png]\n[Image attached for model viewing]",
        tool_result_images=[{"media_type": "image/png", "data": "BASE64DATA"}],
    )


class TestDetectProvider:
    def test_anthropic_variants(self):
        assert _detect_provider("claude-sonnet-4-5") == "anthropic"
        assert _detect_provider("anthropic/claude-3-5-sonnet") == "anthropic"
        assert _detect_provider("anthropic/claude-opus-4") == "anthropic"

    def test_gemini_variants(self):
        assert _detect_provider("gemini/gemini-2.5-pro") == "gemini"
        assert _detect_provider("google/gemini-2.5-flash") == "gemini"
        assert _detect_provider("vertex_ai/gemini-2.5-pro") == "gemini"

    def test_openai_variants(self):
        assert _detect_provider("gpt-4o") == "openai"
        assert _detect_provider("gpt-5-mini") == "openai"
        assert _detect_provider("openai/gpt-4o-mini") == "openai"
        assert _detect_provider("o1-preview") == "openai"
        assert _detect_provider("o3-mini") == "openai"

    def test_unknown_falls_through(self):
        assert _detect_provider("") == "other"
        assert _detect_provider("mistral/large") == "other"


class TestSerializeForProvider:
    def test_anthropic_inlines_image_on_tool_msg(self):
        msg = _make_tool_msg_with_image()
        out = _serialize_messages_for_provider([msg], "claude-sonnet-4-5")
        assert len(out) == 1
        serialized = out[0]
        assert serialized["role"] == "tool"
        assert serialized["tool_call_id"] == "call_1"
        content = serialized["content"]
        assert isinstance(content, list)
        # First block = text, then image_url block
        assert content[0]["type"] == "text"
        assert "Image attached" in content[0]["text"]
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_gemini_inlines_image_on_tool_msg(self):
        msg = _make_tool_msg_with_image()
        out = _serialize_messages_for_provider([msg], "gemini/gemini-2.5-pro")
        assert len(out) == 1
        serialized = out[0]
        assert serialized["role"] == "tool"
        assert isinstance(serialized["content"], list)
        assert any(
            b.get("type") == "image_url" and "base64" in b["image_url"]["url"]
            for b in serialized["content"]
        )

    def test_openai_uses_synthetic_user_workaround(self):
        msg = _make_tool_msg_with_image("call_xyz")
        out = _serialize_messages_for_provider([msg], "gpt-4o")
        assert len(out) == 2
        # First entry: original text-only tool message (goes through
        # to_ll_message, which is an LLMessage object)
        tool_entry = out[0]
        # LLMessage or dict — both expose a 'role' attr/key equivalent
        tool_role = getattr(tool_entry, "role", None) or tool_entry.get("role")  # type: ignore[union-attr]
        assert tool_role == "tool"
        # Second entry: synthetic user message carrying the image
        user_entry = out[1]
        assert isinstance(user_entry, dict)
        assert user_entry["role"] == "user"
        content = user_entry["content"]
        assert any("call_xyz" in b["text"] for b in content if b.get("type") == "text")
        assert any(b.get("type") == "image_url" for b in content)

    def test_other_provider_also_uses_workaround(self):
        msg = _make_tool_msg_with_image()
        out = _serialize_messages_for_provider([msg], "mistral/large")
        # Unknown providers are treated conservatively as "not supporting
        # images in tool results" — same workaround as OpenAI.
        assert len(out) == 2
        assert out[1]["role"] == "user"  # type: ignore[index]

    def test_tool_msg_without_images_unchanged(self):
        msg = Message(
            role="tool",
            tool_call_id="x",
            tool_name="Read",
            content="plain text result",
        )
        out = _serialize_messages_for_provider([msg], "claude-sonnet-4-5")
        assert len(out) == 1
        # No multipart content, no synthetic user msg
        entry = out[0]
        role = getattr(entry, "role", None) or entry.get("role")  # type: ignore[union-attr]
        assert role == "tool"

    def test_non_tool_messages_pass_through(self):
        msgs = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
        ]
        out = _serialize_messages_for_provider(msgs, "gpt-4o")
        assert len(out) == 2


class TestSerializeForResponsesApi:
    def test_tool_msg_with_image_emits_function_output_plus_user(self):
        msg = _make_tool_msg_with_image("call_abc")
        out = _serialize_messages_for_responses_api([msg])
        # function_call_output comes first, then a synthetic user message
        assert any(item.get("type") == "function_call_output" for item in out)
        user_items = [
            item for item in out if item.get("type") == "message" and item.get("role") == "user"
        ]
        assert len(user_items) == 1
        content = user_items[0]["content"]
        assert any(b.get("type") == "input_text" and "call_abc" in b["text"] for b in content)
        assert any(b.get("type") == "input_image" for b in content)

    def test_tool_msg_without_image_unchanged(self):
        msg = Message(
            role="tool",
            tool_call_id="c",
            tool_name="Read",
            content="no image here",
        )
        out = _serialize_messages_for_responses_api([msg])
        # No synthetic user message — just the plain function_call_output
        assert all(
            not (item.get("type") == "message" and item.get("role") == "user") for item in out
        )
