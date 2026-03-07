"""
Ideation Workflow

Partial workflow that only runs IdeationAgent for research ideation.
Useful for generating research ideas independently without running
data analysis or experiments.

For full pipeline, see:
- full_workflow_with_ideation.py: Chains Ideation -> Data -> Experiment
"""

from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, PrivateAttr

from scider.agents import ideation_agent
from scider.agents.ideation_agent.state import IdeationAgentState
from scider.core.brain import Brain
from scider.core.llms import ModelRegistry
from scider.workflows.utils import get_separator


class IdeationWorkflow(BaseModel):
    """
    Ideation Workflow - runs only the IdeationAgent for research ideation.

    This workflow executes:
    1. IdeationAgent - Generates research ideas through literature review

    Usage:
        workflow = IdeationWorkflow(
            user_query="transformer models",
            research_domain="machine learning",
            workspace_path="workspace",
        )
        workflow.run()
        print(workflow.ideation_summary)
    """

    # ==================== INPUT ====================
    user_query: str  # Research topic or query for ideation
    workspace_path: Path
    research_domain: str | None = None  # Optional research domain specification
    recursion_limit: int = 50  # Recursion limit for IdeationAgent

    # Memory directories (optional - if None, will create new Brain session)
    sess_dir: Path | None = None
    long_term_mem_dir: Path | None = None
    project_mem_dir: Path | None = None
    session_name: str | None = None  # Only used if sess_dir is None

    # ==================== INTERNAL STATE ====================
    current_phase: Literal["init", "ideation", "complete", "failed"] = "init"

    # ==================== OUTPUT ====================
    final_status: Literal["success", "failed"] | None = None
    ideation_summary: str = ""
    ideation_papers: list[dict] = []
    research_ideas: list[dict] = []
    idea_novelty_assessments: list[dict] = []  # Per-idea novelty assessments
    novelty_score: float | None = None  # Average novelty score
    novelty_feedback: str | None = None  # Aggregated feedback summary
    error_message: str | None = None

    # Internal: compiled graph (lazy loaded)
    _ideation_agent_graph: object = PrivateAttr(default=None)

    def _ensure_graph(self):
        """Lazily compile ideation agent graph."""
        if self._ideation_agent_graph is None:
            self._ideation_agent_graph = ideation_agent.build().compile()

    def _ensure_ideation_model(self):
        """Ensure ideation model is registered.

        If not registered, fallback to "data" model configuration.
        """
        try:
            ModelRegistry.instance().get_model_params("ideation")
        except ValueError:
            # Fallback to "data" model
            try:
                data_params = ModelRegistry.instance().get_model_params("data")
                ModelRegistry.register(
                    name="ideation",
                    model=data_params["model"],
                    api_key=data_params.get("api_key"),
                    base_url=data_params.get("base_url"),
                )
                logger.debug("Registered ideation model using data model configuration")
            except ValueError:
                logger.warning(
                    "Neither ideation nor data model is registered. Ideation agent may fail."
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

    def run(self) -> "IdeationWorkflow":
        """
        Run the ideation workflow.

        Returns:
            self (for chaining)
        """
        self._ensure_graph()
        self._ensure_ideation_model()
        self._setup_directories()

        logger.info(get_separator())
        logger.info("Starting Ideation Workflow")
        logger.info(get_separator())

        success = self._run_ideation_agent()

        self._finalize(success)

        return self

    def _run_ideation_agent(self) -> bool:
        """
        Run IdeationAgent to generate research ideas.

        Returns:
            True if successful, False if failed
        """
        logger.info("Running IdeationAgent for research ideation")
        self.current_phase = "ideation"

        # Prepare ideation agent state
        ideation_state = IdeationAgentState(
            user_query=self.user_query,
            research_domain=self.research_domain,
        )

        try:
            result = self._ideation_agent_graph.invoke(
                ideation_state,
                {"recursion_limit": self.recursion_limit},
            )
            result_state = IdeationAgentState(**result)

            # Extract results
            self.ideation_summary = result_state.output_summary or ""
            self.ideation_papers = result_state.papers
            self.research_ideas = result_state.research_ideas
            self.idea_novelty_assessments = result_state.idea_novelty_assessments
            self.novelty_score = result_state.novelty_score
            self.novelty_feedback = result_state.novelty_feedback

            logger.info("IdeationAgent completed successfully")
            logger.info(
                f"Generated {len(self.research_ideas)} ideas with avg novelty score: {self.novelty_score:.2f}/10"
                if self.novelty_score is not None
                else f"Generated {len(self.research_ideas)} ideas (no novelty scores)"
            )
            return True

        except Exception as e:
            logger.exception("IdeationAgent failed")
            self.error_message = f"IdeationAgent failed: {e}"
            self.current_phase = "failed"
            return False

    def _finalize(self, success: bool):
        """Finalize the workflow."""
        if success:
            self.current_phase = "complete"
            self.final_status = "success"
        else:
            self.current_phase = "failed"
            self.final_status = "failed"

        logger.info(get_separator())
        logger.info(f"Ideation Workflow completed: {self.final_status}")
        logger.info(get_separator())

    def save_summary(self, path: str | Path | None = None) -> Path:
        """Save the ideation summary to a file."""
        if path is None:
            path = self.workspace_path / "ideation_summary.md"
        path = Path(path)

        summary = f"""# Ideation Summary

## Research Topic
{self.user_query}

## Research Domain
{self.research_domain or 'Not specified'}

## Novelty Assessment
- **Average Score**: {f'{self.novelty_score:.2f}/10' if self.novelty_score is not None else 'N/A'}
- **Ideas Evaluated**: {len(self.idea_novelty_assessments)}

{self._format_per_idea_novelty()}

## Papers Reviewed
{len(self.ideation_papers)} papers reviewed during literature search.

## Status
{self.final_status}

## Ideation Report
> {'\n> '.join(self.ideation_summary.splitlines())}
"""
        path.write_text(summary)
        logger.info(f"Ideation summary saved to {path}")
        return path

    def _format_per_idea_novelty(self) -> str:
        """Format per-idea novelty assessments for display."""
        if not self.idea_novelty_assessments:
            return "No per-idea novelty assessments available."
        lines = []
        for a in self.idea_novelty_assessments:
            lines.append(f"### {a.get('title', 'Unknown')}")
            lines.append(f"- **Score**: {a['novelty_score']:.2f}/10")
            if a.get("feedback"):
                lines.append(f"- **Feedback**: {a['feedback']}")
            lines.append("")
        return "\n".join(lines)


def run_ideation_workflow(
    user_query: str,
    workspace_path: str | Path,
    research_domain: str | None = None,
    recursion_limit: int = 50,
    session_name: str | None = None,
) -> IdeationWorkflow:
    """
    Convenience function to run the ideation workflow.

    Args:
        user_query: Research topic or query for ideation
        workspace_path: Workspace directory for the workflow
        research_domain: Optional research domain specification
        recursion_limit: Recursion limit for IdeationAgent (default=50)
        session_name: Optional custom session name (otherwise uses timestamp)

    Returns:
        IdeationWorkflow: Completed workflow with results

    Example:
        >>> result = run_ideation_workflow(
        ...     user_query="transformer models",
        ...     research_domain="machine learning",
        ...     workspace_path="workspace",
        ... )
        >>> print(result.ideation_summary)
    """
    workflow = IdeationWorkflow(
        user_query=user_query,
        workspace_path=Path(workspace_path),
        research_domain=research_domain,
        recursion_limit=recursion_limit,
        session_name=session_name,
    )
    return workflow.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ideation Workflow - Run IdeationAgent for research ideation",
        prog="python -m scider.workflows.ideation_workflow",
    )
    parser.add_argument("user_query", help="Research topic or query for ideation")
    parser.add_argument("workspace_path", help="Workspace directory for the workflow")
    parser.add_argument(
        "--research-domain",
        default=None,
        help="Optional research domain specification",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=50,
        help="Recursion limit for IdeationAgent (default: 50)",
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="Custom session name (otherwise uses timestamp)",
    )

    args = parser.parse_args()

    result = run_ideation_workflow(
        user_query=args.user_query,
        workspace_path=args.workspace_path,
        research_domain=args.research_domain,
        recursion_limit=args.recursion_limit,
        session_name=args.session_name,
    )

    print("\n" + get_separator())
    print("IDEATION WORKFLOW COMPLETE")
    print(get_separator())
    print(f"\nStatus: {result.final_status}")
    if result.novelty_score is not None:
        print(f"Average Novelty Score: {result.novelty_score:.2f}/10")
    if result.idea_novelty_assessments:
        for a in result.idea_novelty_assessments:
            print(f"  - {a.get('title', 'Unknown')}: {a['novelty_score']:.2f}/10")
    print(f"Papers Reviewed: {len(result.ideation_papers)}")
    print(f"\nIdeation Summary:\n{result.ideation_summary}")
