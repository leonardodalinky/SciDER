from pydantic import model_validator

from scider.core.types import HistoryState, Message, RBankState, ToolsetState


class CriticAgentState(HistoryState, ToolsetState, RBankState):
    # messages to be criticized (input)
    input_msgs: list[Message]
    # current plan of the caller (input)
    plan: str | None = None
    # whether the input messages are from data agent (input)
    is_data_agent: bool = False
    # whether the input messages are from experiment agent (input)
    is_exp_agent: bool = False
    # critics (output)
    critic_msg: Message | None = None

    @model_validator(mode="after")
    def check_agent_source(self):
        if self.is_data_agent and self.is_exp_agent:
            raise ValueError("CriticAgentState: both is_data_agent and is_exp_agent are True")
        if not self.is_data_agent and not self.is_exp_agent:
            raise ValueError("CriticAgentState: both is_data_agent and is_exp_agent are False")
        return self
