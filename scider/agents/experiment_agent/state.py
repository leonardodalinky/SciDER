"""State for the experiment agent."""

from scider.core.code_env import LocalEnv
from scider.core.types import HistoryState


class ExperimentAgentState(HistoryState):
    """State of the experiment agent."""

    workspace: LocalEnv
    data_summary: str
    repo_source: str | None = None
    user_query: str

    # Output
    final_status: str = "pending"
    final_summary: str = ""
    output_summary: str | None = None

    # Intermediate state for UI tracking
    intermediate_state: list[dict] = []
