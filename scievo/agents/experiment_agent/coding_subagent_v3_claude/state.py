from scievo.core.code_env import LocalEnv
from scievo.core.types import HistoryState, ToolsetState


class ClaudeCodingAgentState(ToolsetState, HistoryState):
    """State of the Coding Subagent V3 Claude.

    This agent delegates coding tasks to Claude Agent SDK which has its own
    internal planning mechanism. No external planning is needed.

    Note: No RBankState - memory extraction is not used in this agent.
    """

    # Summary of the data from data agent, providing background info for the coding task (input)
    data_summary: str

    # User's coding task description (input, optional)
    user_query: str | None = None

    # Local environment for the agent (input)
    workspace: LocalEnv

    # Output summary (output)
    output_summary: str | None = None
