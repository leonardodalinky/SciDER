from scievo.core.code_env import LocalEnv
from scievo.core.plan import PlanState
from scievo.core.types import HistoryState, RBankState, ToolsetState


class DataAgentState(ToolsetState, PlanState, HistoryState, RBankState):
    """State of an agent"""

    user_query: str
    # Local environment for the agent
    local_env: LocalEnv

    # talking mode
    talk_mode: bool = False

    @property
    def round(self) -> int:
        return len(self.node_history) - 1
