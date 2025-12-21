from scievo.core.types import HistoryState, ToolsetState


class PaperSearchAgentState(ToolsetState, HistoryState):
    """Minimal state for Paper Search Agent.

    This agent searches for academic papers and datasets using the paper_search and dataset_search toolsets.
    """

    # Input
    user_query: str  # User's search query

    # Output
    papers: list[dict] = []  # Paper search results
    datasets: list[dict] = []  # Dataset search results
    metrics: list[dict] = []  # Extracted metrics from papers
    output_summary: str | None = None  # Final summary
