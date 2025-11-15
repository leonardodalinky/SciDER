from pathlib import Path

from scievo.core.code_env import LocalEnv
from scievo.core.plan import PlanState
from scievo.core.types import HistoryState, ToolsetState


class DataAgentState(ToolsetState, PlanState, HistoryState):
    """State of an agent"""

    user_query: str

    round: int = 0
    # Local environment for the agent
    local_env: LocalEnv
    # session dir (short-term mem storage)
    sess_dir: str | Path
    # long-term mem save dirs (input & output)
    long_term_mem_dir: str | Path
    # project mem save dirs (input & output)
    project_mem_dir: str | Path
    # skip mem extraction for this round
    skip_mem_extraction: bool = False

    # talking mode
    talk_mode: bool = False
