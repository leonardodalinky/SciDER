from scider.core.types import HistoryState


class PaperSearchAgentState(HistoryState):
    """Minimal state for Paper Search Agent.

    This agent searches for academic papers and datasets using paper_search and dataset_search tools.
    Supports iterative query refinement to improve search results.
    """

    # Input
    user_query: str  # User's original search query
    data_summary: str | None = (
        None  # Data analysis summary from data agent (for dataset similarity search)
    )
    current_query: str | None = None  # Current optimized query (for iteration)
    max_search_iterations: int = 3  # Maximum number of search iterations

    # Iteration tracking
    search_iteration: int = 0  # Current search iteration count
    query_history: list[str] = []  # History of queries tried

    # Output
    papers: list[dict] = []  # Paper search results
    datasets: list[dict] = []  # Dataset search results
    metrics: list[dict] = []  # Extracted metrics from papers
    output_summary: str | None = None  # Final summary

    # Approval status (transient, for conditional edge routing)
    approval_status: str | None = None

    # Intermediate states
    intermediate_state: list[dict] = []
