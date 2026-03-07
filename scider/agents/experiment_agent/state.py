"""
State definition for the Experiment Agent.
"""

from typing import Literal

from pydantic import BaseModel

from scider.core.code_env import LocalEnv
from scider.core.types import HistoryState, ToolsetState


class ExperimentAgentState(ToolsetState, HistoryState):
    """State of the high-level Experiment Agent.

    This agent orchestrates the coding -> exec -> summary loop
    with revision capability based on summary analysis.

    Each sub-agent call is stateless - the parent agent only extracts
    relevant outputs without storing full sub-agent states.
    """

    # ==================== INPUT ====================
    # Workspace for the experiment (input, passed to sub-agents)
    workspace: LocalEnv

    # Data summary from the DataAgent (input)
    data_summary: str

    # Repository source - can be a local path or a git URL (input)
    # Note: Git cloning is handled by the coding subagent, not here
    repo_source: str | None = None

    # User's experiment objective/instructions (input)
    user_query: str

    # Maximum number of revision loops allowed (input, default=5)
    max_revisions: int = 5

    # ==================== INTERNAL STATE ====================
    # Current revision loop number (0-indexed)
    current_revision: int = 0

    # Accumulated summaries from each revision loop
    revision_summaries: list[str] = []

    # Current phase in the loop
    current_phase: Literal["init", "coding", "exec", "summary", "analysis", "judge", "complete"] = (
        "init"
    )

    # Intermediate states
    intermediate_state: list[dict] = []

    # ==================== OUTPUT ====================
    # Final experiment result status
    final_status: Literal["success", "failed", "max_revisions_reached"] | None = None

    # Final comprehensive summary
    final_summary: str = ""

    # All execution results from each revision
    all_execution_results: list[dict] = []

    # Detailed results from each loop iteration (for analysis)
    # Each entry contains:
    # {
    #   "coding_summary": str,
    #   "revision": dict,
    # }
    loop_results: list[dict] = []

    # Accumulated analysis of problems and improvements needed
    revision_analysis: str = ""
