"""
Full SciDER Workflow

Complete workflow that chains DataAgent and ExperimentAgent from scratch.
This workflow takes raw data, analyzes it, generates experiment code,
executes it, and produces final metrics.

For partial workflows (e.g., starting from existing data analysis), see:
- data_workflow.py: Only runs DataAgent
- experiment_workflow.py: Only runs ExperimentAgent
"""

import shutil
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, PrivateAttr

from scider.core.constant import override_user_approval
from scider.workflows.data_workflow import DataWorkflow
from scider.workflows.experiment_workflow import ExperimentWorkflow
from scider.workflows.utils import get_separator


class FullWorkflow(BaseModel):
    """
    Full SciDER Workflow - chains DataAgent and ExperimentAgent from scratch.

    This workflow executes:
    1. DataWorkflow - Analyzes input data, produces data_analysis.md
    2. ExperimentWorkflow - Generates code, executes experiments, produces metrics

    Internally uses DataWorkflow and ExperimentWorkflow for better modularity.

    Usage:
        workflow = FullWorkflow(
            data_path="data/data.csv",
            workspace_path="workspace",
            user_query="Train an SVR model",
        )
        workflow.run()
        print(workflow.final_summary)
    """

    # ==================== INPUT ====================
    data_path: Path
    workspace_path: Path
    user_query: str
    repo_source: str | None = None
    max_revisions: int = 5
    data_agent_recursion_limit: int = 100
    experiment_agent_recursion_limit: int = 100
    data_desc: str | None = None  # Optional additional description of the data

    # ==================== INTERNAL STATE ====================
    current_phase: Literal["init", "data_analysis", "experiment", "complete", "failed"] = "init"
    data_summary: str = ""
    data_agent_history: list = []

    # Paper subagent results (from DataWorkflow)
    papers: list[dict] = []
    datasets: list[dict] = []
    metrics: list[dict] = []
    paper_search_summary: str | None = None

    # ==================== OUTPUT ====================
    final_status: Literal["success", "failed", "max_revisions_reached"] | None = None
    final_summary: str = ""
    execution_results: list = []
    error_message: str | None = None

    # Internal: sub-workflows
    _data_workflow: DataWorkflow | None = PrivateAttr(default=None)
    _experiment_workflow: ExperimentWorkflow | None = PrivateAttr(default=None)

    def run(self) -> "FullWorkflow":
        """
        Run the complete workflow: DataWorkflow -> ExperimentWorkflow.

        Returns:
            self (for chaining)
        """

        logger.info(get_separator())
        logger.info("Starting Full SciDER Workflow")
        logger.info(get_separator())

        # Step 1: Run DataWorkflow
        if not self._run_data_phase():
            self._finalize()
            return self

        # Step 2: Run ExperimentWorkflow
        self._run_experiment_phase()

        # Step 3: Finalize
        self._finalize()

        return self

    def _run_data_phase(self) -> bool:
        """
        Run DataWorkflow to analyze the input data.

        Returns:
            True if successful, False if failed
        """
        logger.info("Phase 1: Running DataWorkflow for data analysis")
        self.current_phase = "data_analysis"

        self._data_workflow = DataWorkflow(
            data_path=self.data_path,
            workspace_path=self.workspace_path,
            recursion_limit=self.data_agent_recursion_limit,
            data_desc=self.data_desc,
        )

        try:
            self._data_workflow.run()

            if self._data_workflow.final_status == "success":
                self.data_summary = self._data_workflow.data_summary
                self.data_agent_history = self._data_workflow.data_agent_history
                self._data_workflow.save_summary()
                logger.info("DataWorkflow completed successfully")
                return True
            else:
                self.error_message = self._data_workflow.error_message
                self.current_phase = "failed"
                return False

        except Exception as e:
            logger.exception("DataWorkflow failed")
            self.error_message = f"DataWorkflow failed: {e}"
            self.current_phase = "failed"
            return False

    def _run_experiment_phase(self) -> bool:
        """
        Run ExperimentWorkflow to generate and execute experiments.

        Returns:
            True if successful, False if failed
        """
        logger.info("Phase 2: Running ExperimentWorkflow")
        self.current_phase = "experiment"

        self._experiment_workflow = ExperimentWorkflow(
            workspace_path=self.workspace_path,
            user_query=self.user_query,
            data_summary=self.data_summary,
            repo_source=self.repo_source,
            max_revisions=self.max_revisions,
            recursion_limit=self.experiment_agent_recursion_limit,
        )

        try:
            self._experiment_workflow.run()

            # Extract results from experiment workflow
            self.final_status = self._experiment_workflow.final_status
            self.execution_results = self._experiment_workflow.execution_results
            self.final_summary = self._compose_summary()
            self.current_phase = "complete"

            logger.info(f"ExperimentWorkflow completed: {self.final_status}")
            return True

        except Exception as e:
            logger.exception("ExperimentWorkflow failed")
            self.error_message = f"ExperimentWorkflow failed: {e}"
            self.current_phase = "failed"
            self.final_status = "failed"
            return False

    def _compose_summary(self) -> str:
        """Compose the final summary."""
        exp_summary = (
            self._experiment_workflow.final_summary if self._experiment_workflow else "N/A"
        )
        current_revision = (
            self._experiment_workflow.current_revision if self._experiment_workflow else 0
        )

        return f"""# Full SciDER Workflow Summary

## Data Analysis

{self.data_summary}

---

## Experiment Results

{exp_summary}

---

## Workflow Metadata

- **Data Path**: {self.data_path}
- **Workspace**: {self.workspace_path}
- **Repo Source**: {self.repo_source or 'Not specified'}
- **Final Status**: {self.final_status}
- **Total Revisions**: {current_revision}
"""

    def _finalize(self):
        """Finalize the workflow."""
        logger.info("Finalizing workflow")

        if self.current_phase == "failed":
            self.final_summary = f"# Workflow Failed\n\nError: {self.error_message}"
        elif not self.final_summary:
            self.final_summary = "# Workflow Completed\n\nNo summary available."

        logger.info(get_separator())
        logger.info(f"Workflow completed: {self.final_status}")
        logger.info(get_separator())

    def save_summary(self, path: str | Path | None = None) -> Path:
        """Save the final summary to a file."""
        if path is None:
            path = self.workspace_path / "workflow_summary.md"
        path = Path(path)
        path.write_text(self.final_summary)
        logger.info(f"Summary saved to {path}")
        return path


def run_full_workflow(
    data_path: str | Path,
    workspace_path: str | Path,
    user_query: str,
    repo_source: str | None = None,
    max_revisions: int = 3,
    data_agent_recursion_limit: int = 100,
    experiment_agent_recursion_limit: int = 100,
    data_desc: str | None = None,
    user_approval_enabled: bool = False,
) -> FullWorkflow:
    """
    Convenience function to run the full SciDER workflow.

    Args:
        data_path: Path to the data file or directory to analyze
        workspace_path: Workspace directory for the experiment
        user_query: User's experiment objective
        repo_source: Optional repository source (local path or git URL)
        max_revisions: Maximum revision loops for experiment agent
        data_agent_recursion_limit: Recursion limit for DataAgent (default=100)
        experiment_agent_recursion_limit: Recursion limit for ExperimentAgent (default=100)
        data_desc: Optional additional description of the data

    Returns:
        FullWorkflow: Completed workflow with results

    Example:
        >>> result = run_full_workflow(
        ...     data_path="data/data.csv",
        ...     workspace_path="workspace",
        ...     user_query="Train an SVR model to predict prices",
        ... )
        >>> print(result.final_summary)

    Note:

        These directories are then passed to DataWorkflow and ExperimentWorkflow.
    """
    workflow = FullWorkflow(
        data_path=data_path,
        workspace_path=workspace_path,
        user_query=user_query,
        repo_source=repo_source,
        max_revisions=max_revisions,
        data_agent_recursion_limit=data_agent_recursion_limit,
        experiment_agent_recursion_limit=experiment_agent_recursion_limit,
        data_desc=data_desc,
    )
    with override_user_approval(user_approval_enabled):
        return workflow.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Full SciDER Workflow - Run complete workflow (DataAgent -> ExperimentAgent)",
        prog="python -m scider.workflows.full_workflow",
    )
    parser.add_argument("data_path", help="Path to the data file or directory to analyze")
    parser.add_argument("workspace_path", help="Workspace directory for the experiment")
    parser.add_argument("user_query", help="User's experiment objective")
    parser.add_argument(
        "repo_source",
        nargs="?",
        default=None,
        help="Optional repository source (local path or git URL)",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=5,
        help="Maximum revision loops for ExperimentAgent (default: 5)",
    )
    parser.add_argument(
        "--data-recursion-limit",
        type=int,
        default=100,
        help="Recursion limit for DataAgent (default: 100)",
    )
    parser.add_argument(
        "--experiment-recursion-limit",
        type=int,
        default=100,
        help="Recursion limit for ExperimentAgent (default: 100)",
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="Custom session name (otherwise uses timestamp)",
    )
    parser.add_argument(
        "--data-desc",
        default=None,
        help="Optional additional description of the data",
    )

    args = parser.parse_args()

    result = run_full_workflow(
        data_path=args.data_path,
        workspace_path=args.workspace_path,
        user_query=args.user_query,
        repo_source=args.repo_source,
        max_revisions=args.max_revisions,
        data_agent_recursion_limit=args.data_recursion_limit,
        experiment_agent_recursion_limit=args.experiment_recursion_limit,
        data_desc=args.data_desc,
    )

    print("\n" + get_separator())
    print("FULL WORKFLOW COMPLETE")
    print(get_separator())
    print(f"\nStatus: {result.final_status}")
    print(f"\nFinal Summary:\n{result.final_summary}")
