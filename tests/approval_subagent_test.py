"""Tests for scider.agents.approval_subagent — prompt rendering + parse_verdict."""

from scider.agents.approval_subagent.execute import (
    NODE_APPROVAL_CRITERIA,
    _build_task_prompt,
    parse_verdict_node,
)
from scider.agents.approval_subagent.state import ApprovalSubagentState
from scider.core.types import Message
from scider.prompts import PROMPTS


class TestPromptRendering:
    def test_system_prompt_renders(self):
        s = PROMPTS.approval_subagent.system_prompt.render()
        assert len(s) > 500
        # Must instruct the agent on output format
        assert "verdict" in s
        assert '"approve"' in s and '"reject"' in s

    def test_user_review_criteria_registered(self):
        assert "user_review" in NODE_APPROVAL_CRITERIA
        assert "shipping-gate" in NODE_APPROVAL_CRITERIA["user_review"]

    def test_task_prompt_includes_all_context(self):
        st = ApprovalSubagentState(
            node_name="user_review",
            summary="## Critic: poor — missing files",
            parent_agent="data",
            user_query="analyze housing",
            critic_feedback="CRITICAL: analysis.csv does not exist",
        )
        prompt = _build_task_prompt(st)
        assert "analyze housing" in prompt
        assert "missing files" in prompt
        assert "analysis.csv does not exist" in prompt
        assert "verdict" in prompt  # output format reminder
        assert "shipping-gate" in prompt  # criteria injected
        # parent_agent variable substitution
        assert "data agent" in prompt

    def test_task_prompt_unknown_node_has_fallback(self):
        st = ApprovalSubagentState(node_name="something_new", summary="s")
        prompt = _build_task_prompt(st)
        assert "Judge whether" in prompt  # fallback text


class TestParseVerdict:
    def _state_with_assistant(self, content: str) -> ApprovalSubagentState:
        st = ApprovalSubagentState(node_name="user_review", summary="s")
        st.add_message(Message(role="user", content="q"))
        st.add_message(Message(role="assistant", content=content))
        return st

    def test_parse_reject_with_feedback(self):
        st = self._state_with_assistant(
            "Here is my verdict:\n\n```json\n"
            '{"verdict": "reject", "feedback": "paper.pdf is missing"}\n```'
        )
        parse_verdict_node(st)
        assert st.verdict == "reject"
        assert "paper.pdf" in (st.feedback or "")

    def test_parse_approve(self):
        st = self._state_with_assistant('```json\n{"verdict": "approve", "feedback": ""}\n```')
        parse_verdict_node(st)
        assert st.verdict == "approve"

    def test_parse_raw_json_no_fence(self):
        st = self._state_with_assistant('{"verdict": "approve", "feedback": "ok"}')
        parse_verdict_node(st)
        assert st.verdict == "approve"

    def test_parse_non_json_fail_open(self):
        st = self._state_with_assistant("I think this is fine, no worries")
        parse_verdict_node(st)
        assert st.verdict == "approve"

    def test_parse_invalid_verdict_keyword_fail_open(self):
        st = self._state_with_assistant('{"verdict": "maybe", "feedback": "unsure"}')
        parse_verdict_node(st)
        assert st.verdict == "approve"

    def test_parse_no_assistant_msg_fail_open(self):
        st = ApprovalSubagentState(node_name="user_review", summary="s")
        st.add_message(Message(role="user", content="q"))
        parse_verdict_node(st)
        assert st.verdict == "approve"
