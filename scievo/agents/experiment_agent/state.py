from pathlib import Path

from scievo.core.code_env import LocalEnv
from scievo.core.plan import PlanState
from scievo.core.types import HistoryState, ToolsetState


class ExperimentAgentState(ToolsetState, PlanState, HistoryState):
    """State of ExperimentAgent"""

    user_query: str

    round: int = 0

    # User's instructions
    user_instructions: str = ""

    # Local environment for the agent
    local_env: LocalEnv

    # session dir (mem storage)
    sess_dir: str | Path

    # skip mem extraction for this round
    skip_mem_extraction: bool = False

    # talking mode
    talk_mode: bool = False

    # === Newly added for ExperimentAgent ===
    repo_dir: str | Path | None = None
    readme_text: str | None = None

