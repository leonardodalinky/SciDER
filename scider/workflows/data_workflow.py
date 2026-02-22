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
from scider.core.brain import Brain
from scider.core.code_env import LocalEnv
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

    # Memory directories (optional - if None, will create new Brain session)
    sess_dir: Path | None = None
    long_term_mem_dir: Path | None = None
    project_mem_dir: Path | None = None
    session_name: str | None = None  # Only used if sess_dir is None

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
            self.data_agent_intermediate_state = result_state.intermediate_state
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
        session_name=args.session_name,
    )

    print("\n" + get_separator())
    print("DATA WORKFLOW COMPLETE")
    print(get_separator())
    print(f"\nStatus: {result.final_status}")
    print(f"\nData Summary:\n{result.data_summary}")
