"""Tests for SubagentApprovalHandler — hook decisions, fail-open, env dispatch."""

from unittest.mock import patch

from scider.core import constant
from scider.core.approval import (
    ApprovalContext,
    ApprovalResult,
    AutoApprovalHandler,
    SubagentApprovalHandler,
    get_default_handler,
)


class TestPassthrough:
    def test_unhooked_node_returns_approved(self):
        h = SubagentApprovalHandler()
        resp = h.request_approval("hypo_data_spec", "summary", context=None)
        assert resp.result == ApprovalResult.APPROVED

    def test_hooked_node_without_context_returns_approved(self):
        h = SubagentApprovalHandler()
        resp = h.request_approval("user_review", "summary", context=None)
        assert resp.result == ApprovalResult.APPROVED

    def test_unhooked_node_with_context_returns_approved(self):
        h = SubagentApprovalHandler()
        ctx = ApprovalContext(parent_agent="data")
        resp = h.request_approval("plan_review", "summary", context=ctx)
        assert resp.result == ApprovalResult.APPROVED


class TestSubagentInvocation:
    def test_hooked_with_context_invokes_subagent(self):
        h = SubagentApprovalHandler()
        ctx = ApprovalContext(parent_agent="data", user_query="q", critic_feedback="f")
        with patch(
            "scider.agents.approval_subagent.run_approval_subagent",
            return_value=("approve", None),
        ) as mock_run:
            resp = h.request_approval("user_review", "summary", context=ctx)
        mock_run.assert_called_once()
        assert resp.result == ApprovalResult.APPROVED

    def test_reject_verdict_maps_to_feedback(self):
        h = SubagentApprovalHandler()
        ctx = ApprovalContext(parent_agent="data")
        with patch(
            "scider.agents.approval_subagent.run_approval_subagent",
            return_value=("reject", "files missing"),
        ):
            resp = h.request_approval("user_review", "summary", context=ctx)
        assert resp.result == ApprovalResult.FEEDBACK
        assert "files missing" in (resp.feedback or "")

    def test_subagent_exception_is_fail_open(self):
        h = SubagentApprovalHandler()
        ctx = ApprovalContext(parent_agent="data")
        with patch(
            "scider.agents.approval_subagent.run_approval_subagent",
            side_effect=RuntimeError("model broke"),
        ):
            resp = h.request_approval("user_review", "summary", context=ctx)
        assert resp.result == ApprovalResult.APPROVED


class TestDefaultHandlerDispatch:
    def test_user_approval_on_uses_cli(self):
        with (
            patch.object(constant, "USER_APPROVAL_ENABLED", True),
            patch("scider.core.approval._is_jupyter", return_value=False),
        ):
            h = get_default_handler()
        assert h.__class__.__name__ == "CLIApprovalHandler"

    def test_auto_approval_with_subagent_enabled(self):
        with (
            patch.object(constant, "USER_APPROVAL_ENABLED", False),
            patch.object(constant, "APPROVAL_SUBAGENT_ENABLED", True),
        ):
            h = get_default_handler()
        assert isinstance(h, SubagentApprovalHandler)

    def test_auto_approval_with_subagent_disabled(self):
        with (
            patch.object(constant, "USER_APPROVAL_ENABLED", False),
            patch.object(constant, "APPROVAL_SUBAGENT_ENABLED", False),
        ):
            h = get_default_handler()
        assert isinstance(h, AutoApprovalHandler)


class TestContextKwargBackcompat:
    """Existing handlers must still accept context=None kwarg (signature compat)."""

    def test_cli_handler_accepts_context_kwarg(self):
        # Don't call request_approval (would block). Just ensure the signature accepts it.
        import inspect

        from scider.core.approval import CLIApprovalHandler

        sig = inspect.signature(CLIApprovalHandler().request_approval)
        assert "context" in sig.parameters

    def test_auto_handler_ignores_context(self):
        h = AutoApprovalHandler()
        ctx = ApprovalContext(parent_agent="data")
        resp = h.request_approval("any_node", "summary", context=ctx)
        assert resp.result == ApprovalResult.APPROVED

    def test_jupyter_handler_accepts_context_kwarg(self):
        import inspect

        from scider.core.approval import JupyterApprovalHandler

        sig = inspect.signature(JupyterApprovalHandler().request_approval)
        assert "context" in sig.parameters
