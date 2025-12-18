"""
Full SciEvo Workflow

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

from scievo.core.brain import Brain
from scievo.workflows.data_workflow import DataWorkflow
from scievo.workflows.experiment_workflow import ExperimentWorkflow


def get_separator(margin: int = 4, char: str = "=") -> str:
    """
    Generate a separator that fits the terminal width.

    Args:
        margin: Number of characters to leave as margin (default: 4)
        char: Character to use for separator (default: '=')

    Returns:
        Separator string that fits terminal width
    """
    terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    # Leave margin to be safe and ensure minimum width
    separator_width = max(terminal_width - margin, 10)
    return char * separator_width


class FullWorkflow(BaseModel):
    """
    Full SciEvo Workflow - chains DataAgent and ExperimentAgent from scratch.

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
    session_name: str | None = None  # Optional custom session name
    brain_dir: Path | None = (
        None  # Optional brain directory (uses BRAIN_DIR env if None)
    )

    # ==================== INTERNAL STATE ====================
    current_phase: Literal[
        "init", "data_analysis", "experiment", "complete", "failed"
    ] = "init"
    data_summary: str = ""
    data_agent_history: list = []

    # Brain-managed directories (initialized in _setup_brain)
    sess_dir: Path | None = None
    long_term_mem_dir: Path | None = None
    project_mem_dir: Path | None = None

    # ==================== OUTPUT ====================
    final_status: Literal["success", "failed", "max_revisions_reached"] | None = None
    final_summary: str = ""
    execution_results: list = []
    error_message: str | None = None

    # Internal: sub-workflows
    _data_workflow: DataWorkflow | None = PrivateAttr(default=None)
    _experiment_workflow: ExperimentWorkflow | None = PrivateAttr(default=None)

    def _setup_brain(self):
        """Setup Brain session and memory directories."""
        # Setup workspace
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Get Brain instance (uses brain_dir or BRAIN_DIR env)
        brain = Brain.instance()

        # Create session
        if self.session_name:
            brain_session = Brain.new_session_named(self.session_name)
        else:
            brain_session = Brain.new_session()

        # Set memory directories from Brain
        self.sess_dir = brain_session.session_dir
        self.long_term_mem_dir = brain.brain_dir / "mem_long_term"
        self.project_mem_dir = brain.brain_dir / "mem_project"

        # Ensure memory directories exist
        self.long_term_mem_dir.mkdir(parents=True, exist_ok=True)
        self.project_mem_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Brain session: {self.sess_dir}")
        logger.debug(f"Long-term memory: {self.long_term_mem_dir}")
        logger.debug(f"Project memory: {self.project_mem_dir}")

    def run(self) -> "FullWorkflow":
        """
        Run the complete workflow: DataWorkflow -> ExperimentWorkflow.

        Returns:
            self (for chaining)
        """
        # Step 0: Setup Brain session
        self._setup_brain()

        logger.info(get_separator())
        logger.info("Starting Full SciEvo Workflow")
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
            # Pass Brain-managed directories
            sess_dir=self.sess_dir,
            long_term_mem_dir=self.long_term_mem_dir,
            project_mem_dir=self.project_mem_dir,
        )

        try:
            self._data_workflow.run()

            if self._data_workflow.final_status == "success":
                self.data_summary = self._data_workflow.data_summary
                self.data_agent_history = self._data_workflow.data_agent_history
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
            # Pass Brain-managed directories (for future use in ExperimentAgent)
            sess_dir=self.sess_dir,
            long_term_mem_dir=self.long_term_mem_dir,
            project_mem_dir=self.project_mem_dir,
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
            self._experiment_workflow.final_summary
            if self._experiment_workflow
            else "N/A"
        )
        current_revision = (
            self._experiment_workflow.current_revision
            if self._experiment_workflow
            else 0
        )

        return f"""# Full SciEvo Workflow Summary

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
    max_revisions: int = 5,
    brain_dir: str | Path | None = None,
) -> FullWorkflow:
    """
    Convenience function to run the full SciEvo workflow.

    Args:
        data_path: Path to the data file or directory to analyze
        workspace_path: Workspace directory for the experiment
        user_query: User's experiment objective
        repo_source: Optional repository source (local path or git URL)
        max_revisions: Maximum revision loops for experiment agent
        data_agent_recursion_limit: Recursion limit for DataAgent (default=100)
        experiment_agent_recursion_limit: Recursion limit for ExperimentAgent (default=100)
        session_name: Optional custom session name (otherwise uses timestamp)
        brain_dir: Optional brain directory (uses BRAIN_DIR env if None)

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
        Memory directories are managed by Brain singleton at FullWorkflow level:
        - Session dir: Created via Brain.new_session()
        - Long-term memory: brain_dir/mem_long_term
        - Project memory: brain_dir/mem_project

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
        session_name=session_name,
        brain_dir=Path(brain_dir) if brain_dir else None,
    )
    return workflow.run()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python -m scievo.workflows.full_workflow <data_path> <workspace_path> <user_query> [repo_source]"
        )
        sys.exit(1)

    result = run_full_workflow(
        data_path=sys.argv[1],
        workspace_path=sys.argv[2],
        user_query=sys.argv[3],
        repo_source=sys.argv[4] if len(sys.argv) > 4 else None,
    )

    print("\n" + "=" * 80)
    print("FULL WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"\nStatus: {result.final_status}")
    print(f"\nFinal Summary:\n{result.final_summary}")
