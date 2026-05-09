"""State for the ideation agent."""

from scider.core.types import HistoryState


class IdeationAgentState(HistoryState):
    """State of the ideation agent."""

    user_query: str
    research_domain: str | None = None

    # Output
    research_ideas: list[dict] = []
    # Quality signal for the ideation output. Two regimes:
    # - With idea search OFF: mean of LLM-given 0-10 novelty scores from the report.
    # - With idea search ON: BEST composite score in [0, 1] across the final population.
    # Display logic in build.py picks the right label based on which regime ran.
    idea_score: float | None = None
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
