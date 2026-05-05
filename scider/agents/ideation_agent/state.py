"""State for the ideation agent."""

from scider.core.types import HistoryState


class IdeationAgentState(HistoryState):
    """State of the ideation agent."""

    user_query: str
    research_domain: str | None = None

    # Output
    research_ideas: list[dict] = []
    novelty_score: float | None = None
    output_summary: str | None = None

    # Approval
    approval_status: str = ""  # "approved" or "retry"
    approval_retry_count: int = 0
    max_approval_retries: int = 2

    # Intermediate state for UI tracking
    intermediate_state: list[dict] = []

    # Idea search configuration and output
    idea_search_enabled: bool = True
    max_idea_search_calls: int = 60
    idea_search_result: dict | None = None
    composite_scores: list[float] = []
