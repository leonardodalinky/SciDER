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

    # Intermediate state for UI tracking
    intermediate_state: list[dict] = []
