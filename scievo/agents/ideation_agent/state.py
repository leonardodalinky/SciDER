from scievo.core.types import HistoryState, ToolsetState


class IdeationAgentState(ToolsetState, HistoryState):
    """State for Ideation Agent.

    This agent generates research ideas through literature review using the ideation toolset.
    It searches for papers, analyzes them, and generates novel research directions.
    """

    # Input
    user_query: str  # User's research topic or query
    research_domain: str | None = None  # Optional research domain specification

    # Keyword extraction
    search_keywords: list[str] = []  # Keywords extracted from user query for literature search

    # Literature review
    papers: list[dict] = []  # Papers found during literature search
    analyzed_papers: list[dict] = []  # Papers that have been analyzed

    # Intermediate states
    intermediate_state: list[dict] = []

    # Ideation output
    research_ideas: list[dict] = []  # Generated research ideas
    # Per-idea novelty assessments: each dict has "title", "novelty_score", "feedback", "breakdown"
    idea_novelty_assessments: list[dict] = []
    # Aggregate novelty (average of per-idea scores), None if no ideas
    novelty_score: float | None = None
    novelty_feedback: str | None = None  # Aggregated feedback summary
    output_summary: str | None = None  # Final ideation report
