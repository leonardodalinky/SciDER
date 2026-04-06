"""State for the data agent."""

from scider.core.code_env import LocalEnv
from scider.core.types import HistoryState


class DataAgentState(HistoryState):
    """State of the data analysis agent.

    Simplified state — no rigid phases, no toolsets.
    """

    user_query: str
    workspace: LocalEnv
    data_desc: str | None = None

    # Output
    output_summary: str | None = None

    # Paper search results (populated if agent decides to search)
    papers: list[dict] = []
    datasets: list[dict] = []
    metrics: list[dict] = []

    # Critic review
    critic_feedback: str | None = None
    critic_retry_count: int = 0
    max_critic_retries: int = 2

    # Intermediate state for UI tracking
    intermediate_state: list[dict] = []
