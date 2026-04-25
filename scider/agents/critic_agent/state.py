"""State for the critic agent."""

from scider.core.code_env import LocalEnv
from scider.core.types import HistoryState, Message


class CriticAgentState(HistoryState):
    """State of the critic agent."""

    input_msgs: list[Message]
    plan: str | None = None
    is_data_agent: bool = False
    is_exp_agent: bool = False
    critic_msg: Message | None = None
    intermediate_state: list[dict] = []
    # Parent agent's workspace — forwarded so critic's Bash/Read/etc. run
    # with cwd = parent workspace, not wherever the driver was launched.
    # Without this, verification commands like `ls -R` run in the project
    # root and the critic can't see the workspace files at all.
    workspace: LocalEnv | None = None
