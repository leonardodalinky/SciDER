"""State for the approval subagent."""

from pathlib import Path
from typing import Literal

from scider.core.types import HistoryState


class ApprovalSubagentState(HistoryState):
    """State of the approval subagent."""

    # Inputs
    node_name: str
    summary: str
    title: str = ""
    parent_agent: str = ""
    workspace_dir: Path | None = None
    critic_feedback: str | None = None
    user_query: str = ""

    # Outputs
    verdict: Literal["approve", "reject"] | None = None
    feedback: str | None = None
    intermediate_state: list[dict] = []

    model_config = {"arbitrary_types_allowed": True}
