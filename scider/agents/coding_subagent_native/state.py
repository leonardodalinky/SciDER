"""State for the native coding subagent."""

from scider.core.code_env import LocalEnv
from scider.core.types import HistoryState


class NativeCodingAgentState(HistoryState):
    """State for the native coding subagent.

    Uses the standard query() loop with full tool access.
    Field-compatible with ClaudeCodingAgentState for AgentRegistry integration.
    """

    # Inputs (populated by _build_coding_state)
    data_summary: str
    user_query: str | None = None
    workspace: LocalEnv

    # Output (read by _extract_coding_result)
    output_summary: str | None = None

    # Tracking
    intermediate_state: list[dict] = []
