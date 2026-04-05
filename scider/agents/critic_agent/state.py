"""State for the critic agent."""

from scider.core.types import HistoryState, Message


class CriticAgentState(HistoryState):
    """State of the critic agent."""

    input_msgs: list[Message]
    plan: str | None = None
    is_data_agent: bool = False
    is_exp_agent: bool = False
    critic_msg: Message | None = None
    intermediate_state: list[dict] = []
