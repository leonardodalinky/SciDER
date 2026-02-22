"""
Experiment Workflow

Partial workflow that only runs ExperimentAgent for experiment execution.
Requires a pre-existing data summary (e.g., from DataWorkflow or manual input).
Useful for debugging the experiment phase independently.
"""

import shutil
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, PrivateAttr

from scider.agents import experiment_agent
from scider.agents.experiment_agent.state import ExperimentAgentState
from scider.core.code_env import LocalEnv
from scider.workflows.utils import get_separator


class ExperimentWorkflow(BaseModel):
    """
    Experiment Workflow - runs only the ExperimentAgent.

    This workflow executes:
    1. ExperimentAgent - Generates code, executes experiments, produces metrics

    Requires:
    - data_summary: Either a string containing data analysis, or a path to data_analysis.md

    Usage:
        workflow = ExperimentWorkflow(
            workspace_path="workspace",
            user_query="Train an SVR model",
            data_summary="... analysis from DataAgent ...",
        )
        workflow.run()
        print(workflow.final_summary)
    """

    # ==================== INPUT ====================
    workspace_path: Path
    user_query: str
    data_summary: str  # Can be loaded from file or passed directly
    repo_source: str | None = None
    max_revisions: int = 5
    recursion_limit: int = 100

    # ==================== INTERNAL STATE ====================
    current_phase: Literal["init", "experiment", "complete", "failed"] = "init"

    # ==================== OUTPUT ====================
    final_status: Literal["success", "failed", "max_revisions_reached"] | None = None
    final_summary: str = ""
    execution_results: list = []
    current_revision: int = 0
    error_message: str | None = None
    experiment_agent_intermediate_state: list[dict] = []

    # Internal: compiled graph (lazy loaded)
    _experiment_agent_graph: object = PrivateAttr(default=None)

    def _ensure_graph(self):
        """Lazily compile agent graph."""
        if self._experiment_agent_graph is None:
            self._experiment_agent_graph = experiment_agent.build().compile()

    def _setup_directories(self):
        """Setup workspace directory."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_data_analysis_file(
        cls,
        workspace_path: str | Path,
        user_query: str,
        data_analysis_path: str | Path | None = None,
        repo_source: str | None = None,
        max_revisions: int = 5,
        recursion_limit: int = 100,
    ) -> "ExperimentWorkflow":
        """
        Create ExperimentWorkflow by loading data summary from file.

        Args:
            workspace_path: Workspace directory for the experiment
            user_query: User's experiment objective
            data_analysis_path: Path to data_analysis.md (defaults to workspace/data_analysis.md)
            repo_source: Optional repository source
            max_revisions: Maximum revision loops
            recursion_limit: Recursion limit for ExperimentAgent

        Returns:
            ExperimentWorkflow instance
        """
        workspace_path = Path(workspace_path)

        if data_analysis_path is None:
            data_analysis_path = workspace_path / "data_analysis.md"
        else:
            data_analysis_path = Path(data_analysis_path)

        if not data_analysis_path.exists():
            raise FileNotFoundError(
                f"Data analysis file not found: {data_analysis_path}. "
                "Run DataWorkflow first or provide data_summary directly."
            )

        data_summary = data_analysis_path.read_text()

        return cls(
            workspace_path=workspace_path,
            user_query=user_query,
            data_summary=data_summary,
            repo_source=repo_source,
            max_revisions=max_revisions,
            recursion_limit=recursion_limit,
        )

    def run(self) -> "ExperimentWorkflow":
        """
        Run the experiment workflow.

        Returns:
            self (for chaining)
        """
        self._ensure_graph()
        self._setup_directories()

        logger.info(get_separator())
        logger.info("Starting Experiment Workflow")
        logger.info(get_separator())

        success = self._run_experiment_agent()

        self._finalize(success)

        return self

    def _run_experiment_agent(self) -> bool:
        """
        Run ExperimentAgent to generate and execute experiments.

        Returns:
            True if successful, False if failed
        """
        logger.info("Running ExperimentAgent")
        self.current_phase = "experiment"

        exp_state = ExperimentAgentState(
            workspace=LocalEnv(self.workspace_path),
            data_summary=self.data_summary,
            user_query=self.user_query,
            repo_source=self.repo_source,
            max_revisions=self.max_revisions,
        )

        try:
            result = self._experiment_agent_graph.invoke(
                exp_state,
                {"recursion_limit": self.recursion_limit},
            )
            result_state = ExperimentAgentState(**result)

            # Extract results
            self.final_status = result_state.final_status
            self.execution_results = result_state.all_execution_results
            self.current_revision = result_state.current_revision
            self.experiment_agent_intermediate_state = result_state.intermediate_state
            self.final_summary = self._compose_summary(result_state)
            self.current_phase = "complete"

            logger.info(f"ExperimentAgent completed: {self.final_status}")
            return True

        except Exception as e:
            logger.exception("ExperimentAgent failed")
            self.error_message = f"ExperimentAgent failed: {e}"
            self.current_phase = "failed"
            self.final_status = "failed"
            return False

    def _compose_summary(self, exp_state: ExperimentAgentState) -> str:
        """Compose the final summary."""
        DATA_SUMMARY_LIMITS = 2000
        return f"""\
=== Experiment Workflow Summary ===

====== Data Analysis (Input) ======

{self.data_summary[:DATA_SUMMARY_LIMITS]}{'...' if len(self.data_summary) > DATA_SUMMARY_LIMITS else ''}

---

====== Workflow Metadata ======

- **Workspace**: {self.workspace_path}
- **Repo Source**: {self.repo_source or 'Not specified'}
- **Final Status**: {self.final_status}
- **Total Revisions**: {exp_state.current_revision}

---

====== Experiment Results ======

{exp_state.final_summary}

"""

    def _finalize(self, success: bool):
        """Finalize the workflow."""
        logger.info("Finalizing experiment workflow")

        if not success and not self.final_summary:
            self.final_summary = f"# Experiment Workflow Failed\n\nError: {self.error_message}"

        logger.info(get_separator())
        logger.info(f"Experiment Workflow completed: {self.final_status}")
        logger.info(get_separator())

    def save_summary(self, path: str | Path | None = None) -> Path:
        """Save the final summary to a file."""
        if path is None:
            path = self.workspace_path / "experiment_summary.md"
        path = Path(path)
        path.write_text(self.final_summary)
        logger.info(f"Summary saved to {path}")
        return path


def run_experiment_workflow(
    workspace_path: str | Path,
    user_query: str,
    data_summary: str | None = None,
    data_analysis_path: str | Path | None = None,
    repo_source: str | None = None,
    max_revisions: int = 5,
    recursion_limit: int = 100,
) -> ExperimentWorkflow:
    """
    Convenience function to run the experiment workflow.

    Args:
        workspace_path: Workspace directory for the experiment
        user_query: User's experiment objective
        data_summary: Data analysis text (if not provided, loads from file)
        data_analysis_path: Path to data_analysis.md (defaults to workspace/data_analysis.md)
        repo_source: Optional repository source (local path or git URL)
        max_revisions: Maximum revision loops for experiment agent
        recursion_limit: Recursion limit for ExperimentAgent (default=100)

    Returns:
        ExperimentWorkflow: Completed workflow with results

    Example:
        >>> # Option 1: Load from file
        >>> result = run_experiment_workflow(
        ...     workspace_path="workspace",
        ...     user_query="Train an SVR model to predict prices",
        ... )
        >>>
        >>> # Option 2: Pass data summary directly
        >>> result = run_experiment_workflow(
        ...     workspace_path="workspace",
        ...     user_query="Train an SVR model",
        ...     data_summary="The dataset contains 1000 rows...",
        ... )
        >>> print(result.final_summary)
    """
    if data_summary is not None:
        workflow = ExperimentWorkflow(
            workspace_path=Path(workspace_path),
            user_query=user_query,
            data_summary=data_summary,
            repo_source=repo_source,
            max_revisions=max_revisions,
            recursion_limit=recursion_limit,
        )
    else:
        workflow = ExperimentWorkflow.from_data_analysis_file(
            workspace_path=workspace_path,
            user_query=user_query,
            data_analysis_path=data_analysis_path,
            repo_source=repo_source,
            max_revisions=max_revisions,
            recursion_limit=recursion_limit,
        )

    return workflow.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment Workflow - Run ExperimentAgent for code generation and execution",
        prog="python -m scider.workflows.experiment_workflow",
    )
    parser.add_argument("workspace_path", help="Workspace directory for the workflow")
    parser.add_argument("user_query", help="User's experiment objective")
    parser.add_argument(
        "data_analysis_path",
        nargs="?",
        default=None,
        help="Path to existing data_analysis.md file (optional)",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        help="Recursion limit for ExperimentAgent (default: 100)",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=5,
        help="Maximum revision loops (default: 5)",
    )

    args = parser.parse_args()

    result = run_experiment_workflow(
        workspace_path=args.workspace_path,
        user_query=args.user_query,
        data_analysis_path=args.data_analysis_path,
        recursion_limit=args.recursion_limit,
        max_revisions=args.max_revisions,
    )

    print("\n" + get_separator())
    print("EXPERIMENT WORKFLOW COMPLETE")
    print(get_separator())
    print(f"\nStatus: {result.final_status}")
    print(f"\nFinal Summary:\n{result.final_summary}")
