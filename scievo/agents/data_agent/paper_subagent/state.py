from scievo.core.types import HistoryState, ToolsetState


class PaperSearchAgentState(ToolsetState, HistoryState):
    """Minimal state for Paper Search Agent.

    This agent searches for academic papers using the paper_search toolset.
    """

    # Input
    user_query: str  # User's search query

    # Output
    papers: list[dict] = []  # Search results
    output_summary: str | None = None  # Final summary
