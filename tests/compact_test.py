"""Tests for scider.core.compact — compression pipeline."""

from scider.core.compact import CompactState, _build_persisted_reference, apply_tool_result_budget
from scider.core.types import Message


class TestToolResultBudget:
    def test_small_result_unchanged(self):
        """Results under threshold should not be modified."""
        msg = Message(role="tool", content="short result", tool_name="Glob", tool_call_id="tc1")
        history = [msg]
        state = CompactState()
        result = apply_tool_result_budget(history, state)
        assert result[0].content == "short result"
        assert result[0].persisted_content_path is None

    def test_large_result_persisted(self):
        """Results over threshold should be persisted to disk."""
        large_content = "x" * 60_000  # over 50K default
        msg = Message(role="tool", content=large_content, tool_name="Bash", tool_call_id="tc_large")
        history = [msg]
        state = CompactState()
        result = apply_tool_result_budget(history, state)
        assert "<persisted-output>" in result[0].content
        assert result[0].persisted_content_path is not None
        assert "tc_large" in state.persisted_tool_ids

    def test_read_tool_skipped(self):
        """Read tool has max_result_size_chars=inf, should never be persisted."""
        large_content = "x" * 200_000
        msg = Message(role="tool", content=large_content, tool_name="Read", tool_call_id="tc_read")
        history = [msg]
        state = CompactState()
        result = apply_tool_result_budget(history, state)
        # Should NOT be persisted (Read has inf threshold)
        assert "<persisted-output>" not in result[0].content

    def test_idempotent(self):
        """Running budget twice should not double-persist."""
        large_content = "x" * 60_000
        msg = Message(role="tool", content=large_content, tool_name="Bash", tool_call_id="tc_idem")
        history = [msg]
        state = CompactState()
        apply_tool_result_budget(history, state)
        content_after_first = msg.content
        apply_tool_result_budget(history, state)
        assert msg.content == content_after_first  # unchanged


class TestPersistedReference:
    def test_format(self):
        ref = _build_persisted_reference("/tmp/test.txt", 100_000, "preview text")
        assert "<persisted-output>" in ref
        assert "100,000" in ref
        assert "preview text" in ref
        assert "Do NOT try to fetch" in ref
