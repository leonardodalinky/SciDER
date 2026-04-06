"""Tests for scider.core.types — Message, HistoryState, tool pairing."""

from scider.core.types import HistoryState, Message, _ensure_tool_pairing


class TestMessage:
    def test_basic_message(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.is_meta is False

    def test_meta_message(self):
        msg = Message(role="user", content="ctx", is_meta=True)
        assert msg.is_meta is True

    def test_tool_message(self):
        msg = Message(role="tool", content="result", tool_name="Glob", tool_call_id="tc_1")
        assert msg.tool_name == "Glob"
        assert msg.tool_call_id == "tc_1"

    def test_to_plain_text(self):
        msg = Message(role="assistant", content="answer", agent_sender="data")
        text = msg.to_plain_text()
        assert "answer" in text

    def test_token_count(self):
        msg = Message(role="user", content="hello world")
        assert msg.n_tokens > 0


class TestHistoryState:
    def test_add_message(self):
        state = HistoryState()
        state.add_message(Message(role="user", content="hi"))
        assert len(state.messages) == 1

    def test_total_tokens(self):
        state = HistoryState()
        state.add_message(Message(role="user", content="hello world"))
        assert state.total_tokens > 0

    def test_compact_boundary(self):
        state = HistoryState()
        state.add_message(Message(role="user", content="old msg"))
        state.add_message(Message(role="assistant", content="old reply"))
        # Simulate compaction
        summary_msg = Message(
            role="assistant",
            content="[Summary of previous conversation]",
            is_compact_boundary=True,
        )
        state.compact([summary_msg])
        # messages property should start from boundary
        assert len(state.messages) >= 1
        assert state.messages[0].is_compact_boundary


class TestToolPairing:
    def _make_tool_call(self, tc_id, name="Glob"):
        """Helper to create a mock tool_call object."""
        from unittest.mock import MagicMock

        tc = MagicMock()
        tc.id = tc_id
        tc.function.name = name
        return tc

    def test_matched_pairs(self):
        assistant = Message(role="assistant", content=None)
        assistant.tool_calls = [self._make_tool_call("tc1")]
        tool_result = Message(role="tool", content="ok", tool_call_id="tc1")

        result = _ensure_tool_pairing([assistant, tool_result])
        assert len(result) == 2
        assert result[0].role == "assistant"
        assert result[1].role == "tool"

    def test_orphaned_tool_result(self):
        """Tool result without matching tool_use should be removed."""
        orphan = Message(role="tool", content="orphan", tool_call_id="tc_nonexistent")
        result = _ensure_tool_pairing([orphan])
        assert len(result) == 0

    def test_missing_tool_result(self):
        """Tool use without result should get synthetic result injected."""
        assistant = Message(role="assistant", content=None)
        assistant.tool_calls = [self._make_tool_call("tc1")]

        result = _ensure_tool_pairing([assistant])
        assert len(result) == 2
        assert result[1].role == "tool"
        assert result[1].tool_call_id == "tc1"
        assert "lost" in result[1].content.lower()
