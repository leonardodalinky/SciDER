"""State for the writing agent."""

from pathlib import Path

from scider.core.code_env import LocalEnv
from scider.core.types import HistoryState


class WritingAgentState(HistoryState):
    """State of the paper writing agent.

    The ``workspace`` field is the PAPER workspace — the agent's cwd is set
    there so all its file I/O lands inside it. The SciDER source workspace
    (with the original data / code / experiment artifacts) is exposed read-only
    via ``scider_workspace`` and surfaced in the system-reminder message so
    the agent knows where to cross-reference if needed.
    """

    user_query: str
    workspace: LocalEnv  # paper workspace
    scider_workspace: Path  # source SciDER workspace (read-only)

    # Upstream summaries passed through for reference (already written into
    # paper_workspace/inputs/ by the workflow before the agent starts).
    idea_summary: str = ""
    experiment_summary: str = ""
    data_summary: str = ""

    # Output
    output_summary: str | None = None
    final_tex_path: str | None = None
    final_pdf_path: str | None = None

    # Intermediate state for UI tracking
    intermediate_state: list[dict] = []
