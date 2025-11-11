from pathlib import Path

from scievo.core.code_env import LocalEnv
from scievo.core.plan import PlanState
from scievo.core.types import Message, ToolsetState


class DataAgentState(ToolsetState, PlanState):
    """State of an agent"""

    round: int = 0
    # Local environment for the agent
    local_env: LocalEnv
    # session dir (mem storage)
    sess_dir: str | Path
    # List of messages sent to the agent
    history: list[Message] = []
    # skip mem extraction for this round
    skip_mem_extraction: bool = False

    # talking mode
    talk_mode: bool = False
