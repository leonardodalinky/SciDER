from scievo.core.types import HistoryState, ToolsetState


class IdeationAgentState(ToolsetState, HistoryState):
    """State for Ideation Agent.

    This agent generates research ideas through literature review using the ideation toolset.
    It searches for papers, analyzes them, and generates novel research directions.
    """

    # Input
    user_query: str  # User's research topic or query
    research_domain: str | None = None  # Optional research domain specification

    # Literature review
    papers: list[dict] = []  # Papers found during literature search
    analyzed_papers: list[dict] = []  # Papers that have been analyzed

    # Ideation output
    research_ideas: list[dict] = []  # Generated research ideas
    output_summary: str | None = None  # Final ideation report
