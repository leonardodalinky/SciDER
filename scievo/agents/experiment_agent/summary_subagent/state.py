from scievo.core.code_env import LocalEnv
from scievo.core.types import HistoryState, ToolsetState


class SummaryAgentState(ToolsetState, HistoryState):
    """State of the Summary Agent.

    This agent is responsible for generating comprehensive experiment summaries
    by analyzing conversation history and reading relevant output files.
    """

    # Local environment for the agent (input)
    workspace: LocalEnv

    # Conversation history (input)
    # history: list[dict]   # inherited from HistoryState

    # Output path for the summary file (input, optional)
    output_path: str | None = None

    # Free-form markdown summary text (output)
    # This is human-readable markdown containing the experiment summary
    summary_text: str = ""

    # Intermediate states
    intermediate_state: list[dict] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # add initial toolset
        self.toolsets.append("fs")
