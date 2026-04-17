"""Phase-3 plumbing tests: max_revisions flows from workflow layer into the
agent state's ``max_critic_retries``, which is what ``user_review_node``
actually checks for the early-stop gate.

Prior to this phase ``ExperimentWorkflow.max_revisions`` was stored on the
workflow but never passed to the state, leaving the state default of 2 always
in effect regardless of the caller's choice. This test pins the fix.
"""

from pathlib import Path

import pytest


def _fake_build_with_captured_state(captured: dict):
    """Return a (build, graph) pair that captures the state passed to invoke()."""

    class _Graph:
        def compile(self):
            return self

        def invoke(self, state, _cfg):
            captured["state"] = state
            d = state.model_dump()
            d["intermediate_state"] = []
            d["output_summary"] = "ok"
            d["final_summary"] = "ok"
            d["final_status"] = "success"
            return d

    return _Graph()


class TestExperimentWorkflowMaxRevisions:
    def test_max_revisions_is_passed_to_state(self, tmp_path: Path, monkeypatch):
        from scider.workflows import experiment_workflow as mod

        captured: dict = {}
        monkeypatch.setattr(
            mod.experiment_agent, "build", lambda: _fake_build_with_captured_state(captured)
        )

        wf = mod.ExperimentWorkflow(
            workspace_path=tmp_path / "ws",
            user_query="q",
            data_summary="",
            max_revisions=1,
        )
        wf.run()
        assert captured["state"].max_critic_retries == 1

    def test_default_max_revisions_still_5(self, tmp_path: Path, monkeypatch):
        """Default is 5 at the workflow layer — backward compat."""
        from scider.workflows import experiment_workflow as mod

        captured: dict = {}
        monkeypatch.setattr(
            mod.experiment_agent, "build", lambda: _fake_build_with_captured_state(captured)
        )

        wf = mod.ExperimentWorkflow(
            workspace_path=tmp_path / "ws",
            user_query="q",
            data_summary="",
        )
        wf.run()
        assert captured["state"].max_critic_retries == 5


class TestDataWorkflowMaxRevisions:
    def test_max_revisions_is_passed_to_state(self, tmp_path: Path, monkeypatch):
        from scider.workflows import data_workflow as mod

        captured: dict = {}
        monkeypatch.setattr(
            mod.data_agent, "build", lambda: _fake_build_with_captured_state(captured)
        )
        monkeypatch.setattr("scider.core.hf_dataset.resolve_data_path", lambda p: Path(p))

        wf = mod.DataWorkflow(
            data_path=tmp_path / "in.csv",
            workspace_path=tmp_path / "ws",
            max_revisions=1,
        )
        wf.run()
        assert captured["state"].max_critic_retries == 1

    def test_default_max_revisions_is_2(self, tmp_path: Path, monkeypatch):
        from scider.workflows import data_workflow as mod

        captured: dict = {}
        monkeypatch.setattr(
            mod.data_agent, "build", lambda: _fake_build_with_captured_state(captured)
        )
        monkeypatch.setattr("scider.core.hf_dataset.resolve_data_path", lambda p: Path(p))

        wf = mod.DataWorkflow(
            data_path=tmp_path / "in.csv",
            workspace_path=tmp_path / "ws",
        )
        wf.run()
        assert captured["state"].max_critic_retries == 2


class TestEarlyStopGateWiring:
    """Sanity: when user_approval_enabled=False, the user_review handler is
    SubagentApprovalHandler, which is the component that gives the approval
    subagent its decision authority."""

    def test_override_user_approval_false_switches_to_subagent_handler(self):
        from scider.core import approval, constant
        from scider.core.constant import override_user_approval

        # Reset cached handler so we actually re-detect.
        approval._default_handler = None
        with override_user_approval(False):
            # Both defaults: USER_APPROVAL_ENABLED=False, APPROVAL_SUBAGENT_ENABLED=True
            assert constant.APPROVAL_SUBAGENT_ENABLED is True
            handler = approval._get_handler()
            assert isinstance(handler, approval.SubagentApprovalHandler)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
