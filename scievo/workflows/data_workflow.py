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

from scievo.agents import data_agent
from scievo.agents.data_agent.paper_subagent import build as paper_subagent_build
from scievo.agents.data_agent.paper_subagent.state import PaperSearchAgentState
from scievo.agents.data_agent.state import DataAgentState
from scievo.core.brain import Brain
from scievo.core.code_env import LocalEnv
from scievo.core.llms import ModelRegistry
from scievo.prompts import PROMPTS
from scievo.workflows.utils import get_separator


class DataWorkflow(BaseModel):
    """
    Data Workflow - runs DataAgent for data analysis and Paper Subagent for research.

    This workflow executes:
    1. DataAgent - Analyzes input data, produces data_analysis.md
    2. Paper Subagent - Searches for relevant papers, datasets, and metrics

    Usage:
        workflow = DataWorkflow(
            data_path="data/data.csv",
            workspace_path="workspace",
            user_query="iris classification task",
        )
        workflow.run()
        print(workflow.data_summary)
    """

    # ==================== INPUT ====================
    data_path: Path
    workspace_path: Path
    user_query: str | None = None  # User query for paper subagent search
    recursion_limit: int = 100
    data_desc: str | None = None  # Optional additional description of the data

    # Memory directories (optional - if None, will create new Brain session)
    sess_dir: Path | None = None
    long_term_mem_dir: Path | None = None
    project_mem_dir: Path | None = None
    session_name: str | None = None  # Only used if sess_dir is None

    # ==================== INTERNAL STATE ====================
    current_phase: Literal["init", "data_analysis", "paper_search", "complete", "failed"] = "init"

    # ==================== OUTPUT ====================
    final_status: Literal["success", "failed"] | None = None
    data_summary: str = ""
    data_agent_history: list = []
    error_message: str | None = None

    # Paper subagent results
    papers: list[dict] = []
    datasets: list[dict] = []
    metrics: list[dict] = []
    paper_search_summary: str | None = None

    # Internal: compiled graph (lazy loaded)
    _data_agent_graph: object = PrivateAttr(default=None)
    _paper_subagent_graph: object = PrivateAttr(default=None)

    def _ensure_graph(self):
        """Lazily compile agent graphs."""
        if self._data_agent_graph is None:
            self._data_agent_graph = data_agent.build().compile()
        if self._paper_subagent_graph is None:
            self._paper_subagent_graph = paper_subagent_build().compile()

    def _ensure_paper_subagent_models(self):
        """Ensure paper_search and metric_search models are registered.

        If not registered, fallback to "data" model configuration.
        """
        try:
            ModelRegistry.instance().get_model_params("paper_search")
        except ValueError:
            # Fallback to "data" model
            try:
                data_params = ModelRegistry.instance().get_model_params("data")
                ModelRegistry.register(
                    name="paper_search",
                    model=data_params["model"],
                    api_key=data_params.get("api_key"),
                    base_url=data_params.get("base_url"),
                )
                logger.debug("Registered paper_search model using data model configuration")
            except ValueError:
                logger.warning(
                    "Neither paper_search nor data model is registered. Paper subagent may fail."
                )

        try:
            ModelRegistry.instance().get_model_params("metric_search")
        except ValueError:
            # Fallback to "data" model
            try:
                data_params = ModelRegistry.instance().get_model_params("data")
                ModelRegistry.register(
                    name="metric_search",
                    model=data_params["model"],
                    api_key=data_params.get("api_key"),
                    base_url=data_params.get("base_url"),
                )
                logger.debug("Registered metric_search model using data model configuration")
            except ValueError:
                logger.warning(
                    "Neither metric_search nor data model is registered. Metric extraction may fail."
                )

    def _setup_directories(self):
        """Setup workspace and memory directories.

        If sess_dir is provided (from FullWorkflow), use it.
        Otherwise, create new Brain session (standalone mode).
        """
        # Setup workspace
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Only create Brain session if directories not provided
        if self.sess_dir is None:
            logger.debug("No sess_dir provided, creating new Brain session")
            brain = Brain.instance()
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
        else:
            logger.debug(f"Using provided sess_dir: {self.sess_dir}")

        logger.info(f"Session directory: {self.sess_dir}")
        logger.debug(f"Long-term memory: {self.long_term_mem_dir}")
        logger.debug(f"Project memory: {self.project_mem_dir}")

    def run(self) -> "DataWorkflow":
        """
        Run the data analysis workflow.

        Returns:
            self (for chaining)
        """
        self._ensure_graph()
        self._setup_directories()

        logger.info(get_separator())
        logger.info("Starting Data Workflow")
        logger.info(get_separator())

        success = self._run_data_agent()

        # Run paper subagent if user_query is provided
        if success and self.user_query:
            logger.info("Running Paper Subagent for research search")
            self._run_paper_subagent()

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
        data_query = PROMPTS.data.user_prompt.render(
            dir=str(self.data_path),
            data_desc=self.data_desc,
        )

        # Prepare state
        data_state = DataAgentState(
            workspace=LocalEnv(self.workspace_path),
            sess_dir=Path(self.sess_dir),
            long_term_mem_dir=Path(self.long_term_mem_dir),
            project_mem_dir=Path(self.project_mem_dir),
            user_query=data_query,
            data_desc=self.data_desc,
            talk_mode=False,
        )

        try:
            result = self._data_agent_graph.invoke(
                data_state,
                {"recursion_limit": self.recursion_limit},
            )
            result_state = DataAgentState(**result)

            # Extract data summary from history
            self.data_agent_history = result_state.history
            self.data_summary = self._extract_data_summary(result_state)

            logger.info("DataAgent completed successfully")
            logger.debug(f"Data summary: {len(self.data_summary)} chars")
            return True

        except Exception as e:
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

    def _run_paper_subagent(self) -> bool:
        """
        Run Paper Subagent to search for relevant papers, datasets, and metrics.

        Returns:
            True if successful, False if failed
        """
        logger.info("Running Paper Subagent for research search")
        self.current_phase = "paper_search"

        # Ensure required models are registered (fallback to "data" model if not registered)
        self._ensure_paper_subagent_models()

        try:
            # Prepare paper subagent state
            # Pass data_summary so dataset search can find similar datasets
            paper_state = PaperSearchAgentState(
                user_query=self.user_query,
                data_summary=self.data_summary,  # Pass data analysis summary
            )

            # Invoke paper subagent graph
            result = self._paper_subagent_graph.invoke(paper_state)
            result_state = PaperSearchAgentState(**result)

            # Extract results
            self.papers = result_state.papers
            self.datasets = result_state.datasets
            self.metrics = result_state.metrics
            self.paper_search_summary = result_state.output_summary

            # Integrate paper search results into data_summary
            self._integrate_paper_results()

            logger.info("Paper Subagent completed successfully")
            logger.debug(
                f"Found {len(self.papers)} papers, {len(self.datasets)} datasets, {len(self.metrics)} metrics"
            )
            return True

        except Exception as e:
            logger.exception("Paper Subagent failed")
            # Don't fail the entire workflow if paper search fails
            logger.warning(f"Paper search failed but continuing: {e}")
            return False

    def _integrate_paper_results(self):
        """Integrate paper search results into data_summary."""
        if not self.paper_search_summary:
            return

        # Append paper search summary to data_summary
        paper_section = f"""

---

## Research Context

### Paper Search Summary

{self.paper_search_summary}

### Key Findings

- **Papers Found**: {len(self.papers)}
- **Datasets Found**: {len(self.datasets)}
- **Metrics Extracted**: {len(self.metrics)}

"""
        self.data_summary += paper_section

    def _finalize(self, success: bool):
        """Finalize the workflow."""
        logger.info("Finalizing data workflow")

        if success:
            self.final_status = "success"
            self.current_phase = "complete"
        else:
            self.final_status = "failed"

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
    user_query: str | None = None,
    recursion_limit: int = 100,
    session_name: str | None = None,
    sess_dir: str | Path | None = None,
    long_term_mem_dir: str | Path | None = None,
    project_mem_dir: str | Path | None = None,
    data_desc: str | None = None,
) -> DataWorkflow:
    """
    Convenience function to run the data analysis workflow.

    Args:
        data_path: Path to the data file or directory to analyze
        workspace_path: Workspace directory for the analysis
        user_query: Optional user query for paper subagent search
        recursion_limit: Recursion limit for DataAgent (default=100)
        session_name: Optional custom session name (only used if sess_dir is None)
        sess_dir: Optional session directory (if None, creates new Brain session)
        long_term_mem_dir: Optional long-term memory directory
        project_mem_dir: Optional project memory directory
        data_desc: Optional additional description of the data

    Returns:
        DataWorkflow: Completed workflow with results

    Example:
        >>> # Standalone mode (creates new Brain session)
        >>> result = run_data_workflow(
        ...     data_path="data/data.csv",
        ...     workspace_path="workspace",
        ... )
        >>> print(result.data_summary)

        >>> # With provided directories (e.g., from FullWorkflow)
        >>> result = run_data_workflow(
        ...     data_path="data/data.csv",
        ...     workspace_path="workspace",
        ...     sess_dir=Path("brain/ss_existing"),
        ...     long_term_mem_dir=Path("brain/mem_long_term"),
        ...     project_mem_dir=Path("brain/mem_project"),
        ... )

    Note:
        When sess_dir is None, creates new Brain session automatically:
        - Session dir: Created via Brain.new_session()
        - Long-term memory: brain_dir/mem_long_term
        - Project memory: brain_dir/mem_project
    """
    workflow = DataWorkflow(
        data_path=Path(data_path),
        workspace_path=Path(workspace_path),
        user_query=user_query,
        recursion_limit=recursion_limit,
        sess_dir=Path(sess_dir) if sess_dir else None,
        long_term_mem_dir=Path(long_term_mem_dir) if long_term_mem_dir else None,
        project_mem_dir=Path(project_mem_dir) if project_mem_dir else None,
        session_name=session_name,
        data_desc=data_desc,
    )
    return workflow.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Data Workflow - Run DataAgent for data analysis",
        prog="python -m scievo.workflows.data_workflow",
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
        session_name=args.session_name,
    )

    print("\n" + get_separator())
    print("DATA WORKFLOW COMPLETE")
    print(get_separator())
    print(f"\nStatus: {result.final_status}")
    print(f"\nData Summary:\n{result.data_summary}")
