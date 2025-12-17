from scievo.core.code_env import LocalEnv
from scievo.core.plan import PlanState
from scievo.core.types import HistoryState, RBankState, ToolsetState


class DataAgentState(ToolsetState, PlanState, HistoryState, RBankState):
    """State of an agent"""

    user_query: str
    # Local environment for the agent
    workspace: LocalEnv

    # talking mode
    talk_mode: bool = False
