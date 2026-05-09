"""Phase-2 plumbing tests: WorkspaceInitConfig is accepted by every workflow
constructor, stored on the workflow instance, and threaded down to the
LocalEnv inside each nested agent state.

We do NOT run any agent — that would require LLM keys and real work. We only
verify that the config flows from the top-level API call to the place where
LocalEnv is built. That catches the "forgot to thread" regression that the
phase-2 edit pattern is most prone to.
"""

from pathlib import Path

import pytest

from scider.core.code_env import LocalEnv, WorkspaceInitConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_cfg(venv_path: Path) -> WorkspaceInitConfig:
    """Build a non-default config so we can tell it apart from the default."""
    return WorkspaceInitConfig(
        env_manager="python",
        init_uv=False,
        venv_path=venv_path,
    )


class TestDataWorkflowAcceptsConfig:
    def test_field_is_accepted_and_stored(self, tmp_path: Path):
        from scider.workflows.data_workflow import DataWorkflow

        cfg = _mk_cfg(tmp_path / "v")
        wf = DataWorkflow(
            data_path=tmp_path / "in.csv",
            workspace_path=tmp_path / "ws",
            workspace_init_config=cfg,
        )
        assert wf.workspace_init_config == cfg

    def test_default_is_none_for_backward_compat(self, tmp_path: Path):
        from scider.workflows.data_workflow import DataWorkflow

        wf = DataWorkflow(
            data_path=tmp_path / "in.csv",
            workspace_path=tmp_path / "ws",
        )
        assert wf.workspace_init_config is None


class TestExperimentWorkflowAcceptsConfig:
    def test_field_is_accepted_and_stored(self, tmp_path: Path):
        from scider.workflows.experiment_workflow import ExperimentWorkflow

        cfg = _mk_cfg(tmp_path / "v")
        wf = ExperimentWorkflow(
            workspace_path=tmp_path / "ws",
            user_query="x",
            data_summary="",
            workspace_init_config=cfg,
        )
        assert wf.workspace_init_config == cfg

    def test_default_is_none(self, tmp_path: Path):
        from scider.workflows.experiment_workflow import ExperimentWorkflow

        wf = ExperimentWorkflow(workspace_path=tmp_path / "ws", user_query="x", data_summary="")
        assert wf.workspace_init_config is None


class TestWritingWorkflowAcceptsConfig:
    def test_field_is_accepted_and_stored(self, tmp_path: Path):
        from scider.workflows.writing_workflow import WritingWorkflow

        cfg = _mk_cfg(tmp_path / "v")
        wf = WritingWorkflow(
            scider_workspace_path=tmp_path / "sc",
            idea_summary="i",
            experimental_log="e",
            workspace_init_config=cfg,
        )
        assert wf.workspace_init_config == cfg


class TestIdeationWorkflowAcceptsConfig:
    def test_field_is_accepted_even_though_unused(self, tmp_path: Path):
        from scider.workflows.ideation_workflow import IdeationWorkflow

        cfg = _mk_cfg(tmp_path / "v")
        wf = IdeationWorkflow(
            user_query="x",
            workspace_path=tmp_path / "ws",
            workspace_init_config=cfg,
        )
        # Stored verbatim — the ideation agent doesn't use it, but API
        # uniformity is the point.
        assert wf.workspace_init_config == cfg


class TestHypoDataWorkflowAcceptsAndForwards:
    def test_field_is_accepted_and_stored(self, tmp_path: Path):
        from scider.workflows.hypo_data_workflow import HypoDataWorkflow

        cfg = _mk_cfg(tmp_path / "v")
        wf = HypoDataWorkflow(
            feature_desc="x",
            workspace_path=tmp_path / "ws",
            workspace_init_config=cfg,
        )
        assert wf.workspace_init_config == cfg

    def test_forwards_to_nested_data_workflow(self, tmp_path: Path, monkeypatch):
        """When HypoDataWorkflow.run() proceeds past spec gen, it constructs a
        DataWorkflow — verify our config is passed down."""
        from scider.workflows import hypo_data_workflow as mod

        captured: dict = {}

        class _FakeDW:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.final_status = "success"
                self.data_summary = "ok"
                self.error_message = None

            def run(self):
                return self

        cfg = _mk_cfg(tmp_path / "v")
        wf = mod.HypoDataWorkflow(
            feature_desc="x",
            workspace_path=tmp_path / "ws",
            workspace_init_config=cfg,
        )
        # Skip phases 1/2 entirely by monkeypatching them out.
        monkeypatch.setattr(wf, "_generate_and_approve_spec", lambda: _StubSpec())
        monkeypatch.setattr(mod, "generate_csv_from_spec", lambda *a, **k: tmp_path / "gen.csv")
        monkeypatch.setattr(mod, "DataWorkflow", _FakeDW)

        wf.run()
        assert captured.get("workspace_init_config") == cfg


class _StubSpec:
    """Minimal stand-in for DataGenSpec (we only touch .features)."""

    features: list = []


class TestFullWorkflowForwardsConfig:
    def test_forwards_to_data_and_experiment(self, tmp_path: Path, monkeypatch):
        from scider.workflows import full_workflow as mod

        cfg = _mk_cfg(tmp_path / "v")
        seen: dict = {}

        class _FakeDW:
            def __init__(self, **kwargs):
                seen["data"] = kwargs
                self.final_status = "success"
                self.data_summary = "ok"
                self.data_agent_history = []
                self.error_message = None

            def run(self):
                return self

            def save_summary(self):
                return tmp_path / "summary.md"

        class _FakeEW:
            def __init__(self, **kwargs):
                seen["exp"] = kwargs
                self.final_status = "success"
                self.execution_results = []
                self.final_summary = "ok"
                self.current_revision = 0
                self.error_message = None

            def run(self):
                return self

            def save_summary(self):
                return tmp_path / "exp.md"

        monkeypatch.setattr(mod, "DataWorkflow", _FakeDW)
        monkeypatch.setattr(mod, "ExperimentWorkflow", _FakeEW)

        wf = mod.FullWorkflow(
            data_path=tmp_path / "d.csv",
            workspace_path=tmp_path / "ws",
            user_query="q",
            workspace_init_config=cfg,
        )
        wf.run()
        assert seen["data"]["workspace_init_config"] == cfg
        assert seen["exp"]["workspace_init_config"] == cfg


class TestFullWorkflowWithIdeationForwardsConfig:
    def test_forwards_to_ideation_data_and_experiment(self, tmp_path: Path, monkeypatch):
        from scider.workflows import full_workflow_with_ideation as mod

        cfg = _mk_cfg(tmp_path / "v")
        seen: dict = {}

        class _FakeIW:
            def __init__(self, **kwargs):
                seen["ideation"] = kwargs
                self.final_status = "success"
                self.ideation_summary = "topic"
                self.research_ideas = []
                self.idea_score = None
                self.error_message = None

            def run(self):
                return self

        class _FakeDW:
            def __init__(self, **kwargs):
                seen["data"] = kwargs
                self.final_status = "success"
                self.data_summary = "ok"
                self.error_message = None

            def run(self):
                return self

            def save_summary(self):
                return tmp_path / "sum.md"

        class _FakeEW:
            def __init__(self, **kwargs):
                seen["exp"] = kwargs
                self.final_status = "success"
                self.execution_results = []
                self.final_summary = "ok"
                self.current_revision = 0
                self.error_message = None

            def run(self):
                return self

            def save_summary(self):
                return tmp_path / "exp.md"

        monkeypatch.setattr(mod, "IdeationWorkflow", _FakeIW)
        monkeypatch.setattr(mod, "DataWorkflow", _FakeDW)
        monkeypatch.setattr(mod, "ExperimentWorkflow", _FakeEW)

        wf = mod.FullWorkflowWithIdeation(
            user_query="q",
            workspace_path=tmp_path / "ws",
            data_path=tmp_path / "d.csv",
            run_data_workflow=True,
            run_experiment_workflow=True,
            workspace_init_config=cfg,
        )
        wf.run()
        assert seen["ideation"]["workspace_init_config"] == cfg
        assert seen["data"]["workspace_init_config"] == cfg
        assert seen["exp"]["workspace_init_config"] == cfg


class TestWorkspaceReachesLocalEnvPath:
    """End-to-end sanity: config → DataWorkflow → LocalEnv.

    We don't run the data agent (would need an LLM). We only verify that
    DataAgentState's workspace (a LocalEnv) carries the init_config.
    """

    def test_local_env_receives_config_from_data_workflow(self, tmp_path: Path, monkeypatch):
        from scider.agents.data_agent.state import DataAgentState
        from scider.workflows import data_workflow as mod

        cfg = _mk_cfg(tmp_path / "v")
        (tmp_path / "v" / "bin").mkdir(parents=True)
        captured_state: dict = {}

        class _FakeGraph:
            def invoke(self, state, cfg_kwargs):
                captured_state["state"] = state
                # return a dict that DataAgentState(**dict) can accept
                state_dict = state.model_dump()
                state_dict["intermediate_state"] = []
                state_dict["output_summary"] = "ok"
                return state_dict

        # Monkeypatch build().compile() to return our fake graph.
        monkeypatch.setattr(mod.data_agent, "build", lambda: _FakeBuild())

        # Avoid touching hf_dataset resolution for a non-existent path.
        monkeypatch.setattr("scider.core.hf_dataset.resolve_data_path", lambda p: Path(p))

        wf = mod.DataWorkflow(
            data_path=tmp_path / "in.csv",
            workspace_path=tmp_path / "ws",
            workspace_init_config=cfg,
        )
        wf._data_agent_graph = _FakeGraph()  # avoid build()
        wf._setup_directories()
        # Drive only the agent-running method
        ok = wf._run_data_agent()
        assert ok is True

        workspace: LocalEnv = captured_state["state"].workspace
        assert isinstance(workspace, LocalEnv)
        # The config's non-default markers reached the LocalEnv.
        assert workspace.init_config.env_manager == "python"
        assert workspace.init_config.init_uv is False
        assert workspace.init_config.venv_path is not None
        assert Path(workspace.init_config.venv_path).resolve() == (tmp_path / "v").resolve()


class _FakeBuild:
    """Return self from .compile() — handed to wf._data_agent_graph."""

    def compile(self):
        return self

    def invoke(self, *a, **k):
        raise AssertionError("should be monkeypatched at wf._data_agent_graph level")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
