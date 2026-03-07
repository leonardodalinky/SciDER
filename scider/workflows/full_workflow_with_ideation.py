"""
Full SciDER Workflow with Ideation Agent

Complete workflow that includes IdeationAgent for research ideation.
This workflow can:
1. Run IdeationAgent to generate research ideas based on a research topic
2. Optionally chain with DataWorkflow and ExperimentWorkflow for full research pipeline

For partial workflows, see:
- data_workflow.py: Only runs DataAgent
- experiment_workflow.py: Only runs ExperimentAgent
"""

import shutil
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, PrivateAttr

from scider.core.brain import Brain
from scider.workflows.data_workflow import DataWorkflow
from scider.workflows.experiment_workflow import ExperimentWorkflow
from scider.workflows.ideation_workflow import IdeationWorkflow
from scider.workflows.utils import get_separator


class FullWorkflowWithIdeation(BaseModel):
    """
    Full SciDER Workflow with Ideation Agent.

    This workflow executes:
    1. IdeationAgent - Generates research ideas through literature review
    2. (Optional) DataWorkflow - Analyzes input data, produces data_analysis.md
    3. (Optional) ExperimentWorkflow - Generates code, executes experiments, produces metrics

    Usage:
        # Ideation only
        workflow = FullWorkflowWithIdeation(
            user_query="transformer models",
            research_domain="machine learning",
            workspace_path="workspace",
        )
        workflow.run()
        print(workflow.ideation_summary)

        # Full pipeline: Ideation -> Data -> Experiment
        workflow = FullWorkflowWithIdeation(
            user_query="transformer models",
            research_domain="machine learning",
            workspace_path="workspace",
            data_path="data/data.csv",
            run_data_workflow=True,
            run_experiment_workflow=True,
        )
        workflow.run()
    """

    # ==================== INPUT ====================
    user_query: str  # Research topic or query for ideation
    workspace_path: Path
    research_domain: str | None = None  # Optional research domain specification

    # Optional: Data and Experiment workflows
    data_path: Path | None = None  # Path to data file (if running data workflow)
    run_data_workflow: bool = False  # Whether to run DataWorkflow after ideation
    run_experiment_workflow: bool = False  # Whether to run ExperimentWorkflow after data
    repo_source: str | None = None  # Repository source for experiment workflow
    max_revisions: int = 5  # Maximum revision loops for experiment agent

    # Agent recursion limits
    ideation_agent_recursion_limit: int = 50  # Recursion limit for IdeationAgent
    data_agent_recursion_limit: int = 100  # Recursion limit for DataAgent
    experiment_agent_recursion_limit: int = 100  # Recursion limit for ExperimentAgent

    # Session management
    session_name: str | None = None  # Optional custom session name
    data_desc: str | None = None  # Optional additional description of the data

    # ==================== INTERNAL STATE ====================
    current_phase: Literal[
        "init", "ideation", "data_analysis", "experiment", "complete", "failed"
    ] = "init"

    # Ideation results
    ideation_summary: str = ""
    ideation_papers: list[dict] = []
    idea_novelty_assessments: list[dict] = []  # Per-idea novelty assessments
    novelty_score: float | None = None  # Average novelty score
    novelty_feedback: str | None = None  # Aggregated feedback summary

    # Data and Experiment results (from sub-workflows)
    data_summary: str = ""
    papers: list[dict] = []
    datasets: list[dict] = []
    metrics: list[dict] = []
    paper_search_summary: str | None = None

    # Brain-managed directories (initialized in _setup_brain)
    sess_dir: Path | None = None
    long_term_mem_dir: Path | None = None
    project_mem_dir: Path | None = None

    # ==================== OUTPUT ====================
    final_status: Literal["success", "failed"] | None = None
    final_summary: str = ""
    execution_results: list = []
    error_message: str | None = None

    # Internal: sub-workflows
    _ideation_workflow: IdeationWorkflow | None = PrivateAttr(default=None)
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

    def run(self) -> "FullWorkflowWithIdeation":
        """
        Run the complete workflow: IdeationAgent -> (optional) DataWorkflow -> (optional) ExperimentWorkflow.

        Returns:
            self (for chaining)
        """
        # Step 0: Setup Brain session
        self._setup_brain()

        logger.info(get_separator())
        logger.info("Starting Full SciDER Workflow with Ideation")
        logger.info(get_separator())

        # Step 1: Run IdeationAgent
        if not self._run_ideation_phase():
            self._finalize()
            return self

        # Step 2: (Optional) Run DataWorkflow
        if self.run_data_workflow:
            if not self._run_data_phase():
                self._finalize()
                return self

        # Step 3: (Optional) Run ExperimentWorkflow
        if self.run_experiment_workflow:
            self._run_experiment_phase()

        # Step 4: Finalize
        self._finalize()

        return self

    def _run_ideation_phase(self) -> bool:
        """
        Run IdeationWorkflow to generate research ideas.

        Returns:
            True if successful, False if failed
        """
        logger.info("Phase 1: Running IdeationWorkflow for research ideation")
        self.current_phase = "ideation"

        self._ideation_workflow = IdeationWorkflow(
            user_query=self.user_query,
            workspace_path=self.workspace_path,
            research_domain=self.research_domain,
            recursion_limit=self.ideation_agent_recursion_limit,
            # Pass Brain-managed directories
            sess_dir=self.sess_dir,
            long_term_mem_dir=self.long_term_mem_dir,
            project_mem_dir=self.project_mem_dir,
        )

        try:
            self._ideation_workflow.run()

            if self._ideation_workflow.final_status == "success":
                self.ideation_summary = self._ideation_workflow.ideation_summary
                self.ideation_papers = self._ideation_workflow.ideation_papers
                self.idea_novelty_assessments = self._ideation_workflow.idea_novelty_assessments
                self.novelty_score = self._ideation_workflow.novelty_score
                self.novelty_feedback = self._ideation_workflow.novelty_feedback
                logger.info("IdeationWorkflow completed successfully")
                return True
            else:
                self.error_message = self._ideation_workflow.error_message
                self.current_phase = "failed"
                return False

        except Exception as e:
            logger.exception("IdeationWorkflow failed")
            self.error_message = f"IdeationWorkflow failed: {e}"
            self.current_phase = "failed"
            return False

    def _run_data_phase(self) -> bool:
        """
        Run DataWorkflow to analyze the input data.

        Returns:
            True if successful, False if failed
        """
        if not self.data_path:
            logger.warning("run_data_workflow is True but data_path is not provided")
            return False

        logger.info("Phase 2: Running DataWorkflow for data analysis")
        self.current_phase = "data_analysis"

        self._data_workflow = DataWorkflow(
            data_path=self.data_path,
            workspace_path=self.workspace_path,
            recursion_limit=self.data_agent_recursion_limit,
            data_desc=self.data_desc,
            # Pass Brain-managed directories
            sess_dir=self.sess_dir,
            long_term_mem_dir=self.long_term_mem_dir,
            project_mem_dir=self.project_mem_dir,
        )

        try:
            self._data_workflow.run()

            if self._data_workflow.final_status == "success":
                self.data_summary = self._data_workflow.data_summary
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
        logger.info("Phase 3: Running ExperimentWorkflow")
        self.current_phase = "experiment"

        self._experiment_workflow = ExperimentWorkflow(
            workspace_path=self.workspace_path,
            user_query=self.user_query,
            data_summary=self.data_summary,
            repo_source=self.repo_source,
            max_revisions=self.max_revisions,
            recursion_limit=self.experiment_agent_recursion_limit,
            # Pass Brain-managed directories
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
        novelty_display = ""
        if self.idea_novelty_assessments:
            novelty_lines = []
            for a in self.idea_novelty_assessments:
                novelty_lines.append(
                    f"- **{a.get('title', 'Unknown')}**: {a['novelty_score']:.2f}/10"
                )
            novelty_display = "\n".join(novelty_lines)
        elif self.novelty_score is not None:
            novelty_display = f"{self.novelty_score:.2f}/10"
        else:
            novelty_display = "N/A"

        ideation_section = f"""## Research Ideation

{self.ideation_summary}

**Average Novelty Score**: {f'{self.novelty_score:.2f}/10' if self.novelty_score is not None else 'N/A'}
**Papers Reviewed**: {len(self.ideation_papers)}

### Per-Idea Novelty
{novelty_display}
"""

        data_section = ""
        if self.run_data_workflow and self.data_summary:
            data_section = f"""

---

## Data Analysis

{self.data_summary}
"""

        experiment_section = ""
        if self.run_experiment_workflow and self._experiment_workflow:
            exp_summary = (
                self._experiment_workflow.final_summary if self._experiment_workflow else "N/A"
            )
            experiment_section = f"""

---

## Experiment Results

{exp_summary}
"""

        return f"""# Full SciDER Workflow with Ideation Summary

{ideation_section}{data_section}{experiment_section}

---

## Workflow Metadata

- **Research Topic**: {self.user_query}
- **Research Domain**: {self.research_domain or 'Not specified'}
- **Workspace**: {self.workspace_path}
- **Final Status**: {self.final_status}
"""

    def _finalize(self):
        """Finalize the workflow."""
        logger.info("Finalizing workflow")

        if self.current_phase == "failed":
            self.final_summary = f"# Workflow Failed\n\nError: {self.error_message}"
            self.final_status = "failed"
        elif not self.final_summary:
            self.final_summary = self._compose_summary()
            self.final_status = "success"

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


def run_full_workflow_with_ideation(
    user_query: str,
    workspace_path: str | Path,
    research_domain: str | None = None,
    data_path: str | Path | None = None,
    run_data_workflow: bool = False,
    run_experiment_workflow: bool = False,
    repo_source: str | None = None,
    max_revisions: int = 5,
    ideation_agent_recursion_limit: int = 50,
    data_agent_recursion_limit: int = 100,
    experiment_agent_recursion_limit: int = 100,
    session_name: str | None = None,
    data_desc: str | None = None,
) -> FullWorkflowWithIdeation:
    """
    Convenience function to run the full SciDER workflow with ideation.

    Args:
        user_query: Research topic or query for ideation
        workspace_path: Workspace directory for the workflow
        research_domain: Optional research domain specification
        data_path: Optional path to data file (required if run_data_workflow=True)
        run_data_workflow: Whether to run DataWorkflow after ideation
        run_experiment_workflow: Whether to run ExperimentWorkflow after data
        repo_source: Optional repository source (local path or git URL)
        max_revisions: Maximum revision loops for experiment agent
        ideation_agent_recursion_limit: Recursion limit for IdeationAgent (default=50)
        data_agent_recursion_limit: Recursion limit for DataAgent (default=100)
        experiment_agent_recursion_limit: Recursion limit for ExperimentAgent (default=100)
        session_name: Optional custom session name (otherwise uses timestamp)
        data_desc: Optional additional description of the data

    Returns:
        FullWorkflowWithIdeation: Completed workflow with results

    Example:
        >>> # Ideation only
        >>> result = run_full_workflow_with_ideation(
        ...     user_query="transformer models",
        ...     research_domain="machine learning",
        ...     workspace_path="workspace",
        ... )
        >>> print(result.ideation_summary)

        >>> # Full pipeline: Ideation -> Data -> Experiment
        >>> result = run_full_workflow_with_ideation(
        ...     user_query="transformer models",
        ...     research_domain="machine learning",
        ...     workspace_path="workspace",
        ...     data_path="data/data.csv",
        ...     run_data_workflow=True,
        ...     run_experiment_workflow=True,
        ... )
        >>> print(result.final_summary)
    """
    workflow = FullWorkflowWithIdeation(
        user_query=user_query,
        workspace_path=workspace_path,
        research_domain=research_domain,
        data_path=Path(data_path) if data_path else None,
        run_data_workflow=run_data_workflow,
        run_experiment_workflow=run_experiment_workflow,
        repo_source=repo_source,
        max_revisions=max_revisions,
        ideation_agent_recursion_limit=ideation_agent_recursion_limit,
        data_agent_recursion_limit=data_agent_recursion_limit,
        experiment_agent_recursion_limit=experiment_agent_recursion_limit,
        session_name=session_name,
        data_desc=data_desc,
    )
    return workflow.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Full SciDER Workflow with Ideation - Run ideation agent and optionally data/experiment workflows",
        prog="python -m scider.workflows.full_workflow_with_ideation",
    )
    parser.add_argument("user_query", help="Research topic or query for ideation")
    parser.add_argument("workspace_path", help="Workspace directory for the workflow")
    parser.add_argument(
        "--research-domain",
        default=None,
        help="Optional research domain specification",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to data file (required if --run-data-workflow is set)",
    )
    parser.add_argument(
        "--run-data-workflow",
        action="store_true",
        help="Run DataWorkflow after ideation",
    )
    parser.add_argument(
        "--run-experiment-workflow",
        action="store_true",
        help="Run ExperimentWorkflow after data workflow",
    )
    parser.add_argument(
        "--repo-source",
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
        "--ideation-recursion-limit",
        type=int,
        default=50,
        help="Recursion limit for IdeationAgent (default: 50)",
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

    result = run_full_workflow_with_ideation(
        user_query=args.user_query,
        workspace_path=args.workspace_path,
        research_domain=args.research_domain,
        data_path=args.data_path,
        run_data_workflow=args.run_data_workflow,
        run_experiment_workflow=args.run_experiment_workflow,
        repo_source=args.repo_source,
        max_revisions=args.max_revisions,
        ideation_agent_recursion_limit=args.ideation_recursion_limit,
        data_agent_recursion_limit=args.data_recursion_limit,
        experiment_agent_recursion_limit=args.experiment_recursion_limit,
        session_name=args.session_name,
        data_desc=args.data_desc,
    )

    print("\n" + get_separator())
    print("FULL WORKFLOW WITH IDEATION COMPLETE")
    print(get_separator())
    print(f"\nStatus: {result.final_status}")
    print(f"\nFinal Summary:\n{result.final_summary}")
