from litellm import Message as LLMessage
from pydantic import BaseModel

from .code_env import LocalEnv


class Message(LLMessage):
    # --- LLMessage fields ---
    # content: Optional[str]
    # role: Literal["assistant", "user", "system", "tool", "function"]
    # tool_calls: Optional[List[ChatCompletionMessageToolCall]]
    # function_call: Optional[FunctionCall]
    # audio: Optional[ChatCompletionAudioResponse] = None
    # images: Optional[List[ImageURLListItem]] = None
    # reasoning_content: Optional[str] = None
    # thinking_blocks: Optional[
    #     List[Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]]
    # ] = None
    # provider_specific_fields: Optional[Dict[str, Any]] = Field(
    #     default=None, exclude=True
    # )
    # annotations: Optional[List[ChatCompletionAnnotation]] = None

    # --- custom fields ---
    llm_sender: str | None = None
    agent_sender: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None

    @classmethod
    def from_ll_message(cls, msg: LLMessage) -> "Message":
        o = cls(**msg.__dict__)
        return o

    @property
    def total_tokens(self) -> int | None:
        if self.completion_tokens is None and self.prompt_tokens is None:
            return None
        return (self.completion_tokens or 0) + (self.prompt_tokens or 0)


class GraphState(BaseModel):
    """State of the graph"""

    agents: dict[str, "AgentState"]


class AgentState(BaseModel):
    """State of an agent"""

    # Local environment for the agent
    local_env: LocalEnv
    # List of toolsets available to the agent
    toolsets: list[str] = ["noop"]
    # List of messages sent to the agent
    data_msgs: list[Message] = []
