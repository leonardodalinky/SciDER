"""Tests for AskUserQuestionTool behavior under auto-approval.

Under USER_APPROVAL_ENABLED=false, the tool must return explicit
"non-interactive mode" strings instead of fake answers so the agent
knows nobody answered and makes its own decision.
"""

import json
from unittest.mock import patch

from scider.core import constant
from scider.tools.user.ask_user import AskUserQuestionTool


class _FakeContext:
    """Minimal stand-in for ToolContext."""

    def __init__(self):
        self.extra = {}
        self.agent_state = None


class TestAutoApprovalBehavior:
    def test_options_question_auto_mode_signals_non_interactive(self):
        with patch.object(constant, "USER_APPROVAL_ENABLED", False):
            out = AskUserQuestionTool().call(
                _FakeContext(),
                questions=[
                    {
                        "question": "Which dataset?",
                        "options": [
                            {"label": "Option A", "description": "first"},
                            {"label": "Option B", "description": "second"},
                        ],
                    }
                ],
            )
        data = json.loads(out)
        answer = data["answers"]["Which dataset?"]
        assert "Non-interactive mode" in answer
        assert "Option A" in answer  # default first option surfaced
        assert "make a decision" in answer.lower() or "execute" in answer.lower()

    def test_open_ended_question_auto_mode_signals_non_interactive(self):
        with patch.object(constant, "USER_APPROVAL_ENABLED", False):
            out = AskUserQuestionTool().call(
                _FakeContext(),
                questions=[{"question": "What approach should I use?", "options": []}],
            )
        data = json.loads(out)
        answer = data["answers"]["What approach should I use?"]
        assert "Non-interactive mode" in answer
        assert "make a decision" in answer.lower() or "execute" in answer.lower()

    def test_interactive_mode_options_returns_plain_label(self):
        """When user_approval is ON and a real handler returns approved+selected_index=0,
        the answer should be just the option label (legacy behavior)."""
        from scider.core.approval import ApprovalResponse, ApprovalResult

        with (
            patch.object(constant, "USER_APPROVAL_ENABLED", True),
            patch("scider.core.approval._get_handler") as mock_get,
        ):
            handler = mock_get.return_value
            handler.request_approval_with_selection.return_value = ApprovalResponse(
                result=ApprovalResult.APPROVED, selected_index=0
            )
            out = AskUserQuestionTool().call(
                _FakeContext(),
                questions=[
                    {
                        "question": "Which dataset?",
                        "options": [
                            {"label": "Option A", "description": "first"},
                            {"label": "Option B", "description": "second"},
                        ],
                    }
                ],
            )
        data = json.loads(out)
        assert data["answers"]["Which dataset?"] == "Option A"

    def test_interactive_mode_open_ended_returns_legacy_placeholder(self):
        from scider.core.approval import ApprovalResponse, ApprovalResult

        with (
            patch.object(constant, "USER_APPROVAL_ENABLED", True),
            patch("scider.core.approval._get_handler") as mock_get,
        ):
            handler = mock_get.return_value
            handler.request_approval.return_value = ApprovalResponse(
                result=ApprovalResult.APPROVED, feedback=None
            )
            out = AskUserQuestionTool().call(
                _FakeContext(),
                questions=[{"question": "What approach?", "options": []}],
            )
        data = json.loads(out)
        assert data["answers"]["What approach?"] == "(approved without specific answer)"
