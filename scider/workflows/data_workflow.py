"""
Data Workflow

Partial workflow that only runs DataAgent for data analysis.
Useful for debugging the data analysis phase independently.
"""

import shutil
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, PrivateAttr

from scider.agents import data_agent
from scider.agents.data_agent.state import DataAgentState
from scider.core.code_env import LocalEnv, WorkspaceInitConfig
from scider.core.constant import override_user_approval
from scider.prompts import PROMPTS
from scider.workflows.utils import get_separator


class DataWorkflow(BaseModel):
    """
    Data Workflow - runs only the DataAgent for data analysis.

    This workflow executes:
    1. DataAgent - Analyzes input data, produces data_analysis.md

    Usage:
        workflow = DataWorkflow(
            data_path="data/data.csv",
            workspace_path="workspace",
        )
        workflow.run()
        print(workflow.data_summary)
    """

    # ==================== INPUT ====================
    data_path: Path
    workspace_path: Path
    recursion_limit: int = 100
    data_desc: str | None = None  # Optional additional description of the data
    # Hard upper bound on critic/approval retry loops. The approval subagent
    # may still early-stop before hitting this (that's the whole point).
    max_revisions: int = 2
    # None → LocalEnv's default (uv-managed, auto `uv init`). Override to
    # point agents at a prebuilt venv or to skip `uv init` in the workspace.
    workspace_init_config: WorkspaceInitConfig | None = None

    # ==================== INTERNAL STATE ====================
    current_phase: Literal["init", "data_analysis", "complete", "failed"] = "init"

    # ==================== OUTPUT ====================
    final_status: Literal["success", "failed"] | None = None
    data_summary: str = ""
    data_agent_history: list = []
    data_agent_intermediate_state: list[dict] = []
    error_message: str | None = None

    # Internal: compiled graph (lazy loaded)
    _data_agent_graph: object = PrivateAttr(default=None)

    def _ensure_graph(self):
        """Lazily compile agent graph."""
        if self._data_agent_graph is None:
            self._data_agent_graph = data_agent.build().compile()

    def _setup_directories(self):
        """Setup workspace directory."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)

    def run(self) -> "DataWorkflow":
        """
        Run the data analysis workflow.

        Returns:
            self (for chaining)
        """
        from scider.core.hf_dataset import resolve_data_path

        self.data_path = resolve_data_path(self.data_path)

        self._ensure_graph()
        self._setup_directories()

        logger.info(get_separator())
        logger.info("Starting Data Workflow")
        logger.info(get_separator())

        success = self._run_data_agent()

        self._finalize(success)

        return self

    def _run_data_agent(self) -> bool:
        """
        Run DataAgent to analyze the input data.

        Returns:
            True if successful, False if failed
        """
        logger.info("Running DataAgent for data analysis")
        self.current_phase = "data_analysis"

        # Construct query for data analysis
        data_query = f"Analyze the data at {self.data_path}."
        if self.data_desc:
            data_query += f"\n\nAdditional context: {self.data_desc}"

        # Prepare state
        data_state = DataAgentState(
            workspace=LocalEnv(self.workspace_path, init_config=self.workspace_init_config),
            user_query=data_query,
            data_desc=self.data_desc,
            max_critic_retries=self.max_revisions,
        )

        from scider.workflows.history_export import capture_messages

        with capture_messages() as captured:
            try:
                result = self._data_agent_graph.invoke(
                    data_state,
                    {"recursion_limit": self.recursion_limit},
                )
                result_state = DataAgentState(**result)

                # Prefer `captured` (full pre-compact history via listener) over
                # result_state.history, which compact() would have truncated.
                self.data_agent_history = list(captured)
                self.data_agent_intermediate_state = result_state.intermediate_state
                self.data_summary = self._extract_data_summary(result_state)

                logger.info("DataAgent completed successfully")
                logger.debug(f"Data summary: {len(self.data_summary)} chars")
                return True

            except Exception as e:
                # On failure, fall back to captured messages so debugging info
                # is preserved even when graph.invoke() raised mid-run.
                self.data_agent_history = list(captured)
                logger.exception("DataAgent failed")
                self.error_message = f"DataAgent failed: {e}"
                self.current_phase = "failed"
                return False

    def _extract_data_summary(self, result_state: DataAgentState) -> str:
        """Extract data summary from DataAgent state."""
        # First try to read from output_summary field
        if result_state.output_summary:
            return result_state.output_summary

        # Fallback: try to read saved analysis.md file
        analysis_file = self.workspace_path / "analysis.md"
        if analysis_file.exists():
            return analysis_file.read_text()

        raise RuntimeError("Data analysis completed but no summary was generated.")

    def _finalize(self, success: bool):
        """Finalize the workflow."""
        logger.info("Finalizing data workflow")

        if success:
            self.final_status = "success"
            self.current_phase = "complete"
        else:
            self.final_status = "failed"

        # Save conversation history for debugging
        if self.data_agent_history:
            from scider.workflows.history_export import save_conversation_history

            try:
                save_conversation_history(
                    self.data_agent_history,
                    self.workspace_path / "data_agent_history.json",
                    agent_name="data",
                )
            except Exception as e:
                logger.warning("Failed to save data agent history: {}", e)

        logger.info(get_separator())
        logger.info(f"Data Workflow completed: {self.final_status}")
        logger.info(get_separator())

    def save_summary(self, path: str | Path | None = None) -> Path:
        """Save the data summary to a file."""
        if path is None:
            path = self.workspace_path / "data_analysis.md"
        path = Path(path)
        path.write_text(self.data_summary)
        logger.info(f"Data summary saved to {path}")
        return path


def run_data_workflow(
    data_path: str | Path,
    workspace_path: str | Path,
    recursion_limit: int = 100,
    data_desc: str | None = None,
    user_approval_enabled: bool = False,
    max_revisions: int = 2,
    workspace_init_config: WorkspaceInitConfig | None = None,
) -> DataWorkflow:
    """
    Convenience function to run the data analysis workflow.

    Args:
        data_path: Path to the data file or directory to analyze
        workspace_path: Workspace directory for the analysis
        recursion_limit: Recursion limit for DataAgent (default=100)
        data_desc: Optional additional description of the data
        max_revisions: Hard upper bound on critic/approval retries. The
            approval subagent may early-stop before this when the output
            looks good enough.
        workspace_init_config: Override LocalEnv init behaviour (uv init, PATH
            injection, env manager). Leave ``None`` for historical defaults.

    Returns:
        DataWorkflow: Completed workflow with results
    """
    workflow = DataWorkflow(
        data_path=Path(data_path),
        workspace_path=Path(workspace_path),
        recursion_limit=recursion_limit,
        data_desc=data_desc,
        max_revisions=max_revisions,
        workspace_init_config=workspace_init_config,
    )
    with override_user_approval(user_approval_enabled):
        return workflow.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Data Workflow - Run DataAgent for data analysis",
        prog="python -m scider.workflows.data_workflow",
    )
    parser.add_argument("data_path", help="Path to the data file or directory to analyze")
    parser.add_argument("workspace_path", help="Workspace directory for the workflow")
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        help="Recursion limit for DataAgent (default: 100)",
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="Custom session name (otherwise uses timestamp)",
    )

    args = parser.parse_args()

    result = run_data_workflow(
        data_path=args.data_path,
        workspace_path=args.workspace_path,
        recursion_limit=args.recursion_limit,
    )

    print("\n" + get_separator())
    print("DATA WORKFLOW COMPLETE")
    print(get_separator())
    print(f"\nStatus: {result.final_status}")
    print(f"\nData Summary:\n{result.data_summary}")
