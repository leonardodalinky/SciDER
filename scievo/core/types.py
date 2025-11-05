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

    __CUSTOM_FIELDS__ = [
        "llm_sender",
        "agent_sender",
        "tool_call_id",
        "tool_name",
        "completion_tokens",
        "prompt_tokens",
        "hidden",
    ]

    # --- custom fields ---
    llm_sender: str | None = None
    agent_sender: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    # whether the message is visible to the llms
    hidden: bool | None = None

    @classmethod
    def from_ll_message(cls, msg: LLMessage) -> "Message":
        o = cls(**msg.__dict__)
        return o

    @property
    def total_tokens(self) -> int | None:
        if self.completion_tokens is None and self.prompt_tokens is None:
            return None
        return (self.completion_tokens or 0) + (self.prompt_tokens or 0)

    def to_ll_message(self) -> LLMessage | dict:
        return LLMessage(**self.model_dump(exclude=fields_to_exclude))

    def to_ll_response_message(
        self,
    ) -> list[dict]:
        if self.role == "tool":
            # Only used in OpenAI response API
            return [
                {
                    "type": "function_call_output",
                    "call_id": self.tool_call_id,
                    "output": self.content,
                }
            ]

        fields_to_exclude = self.__CUSTOM_FIELDS__.copy()
        fields_to_exclude.append("tool_calls")

        ret = []

        if self.content is not None:
            ret.append(
                {
                    "type": "message",
                    "role": self.role,
                    "content": self.content,
                }
            )

        if self.tool_calls and len(self.tool_calls) > 0:
            for tool_call in self.tool_calls:
                ret.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                )

        return ret

    def to_plain_text(self) -> str:
        return f"""\
Role: {self.role}
Agent Sender: {self.agent_sender}
Tool Name: {self.tool_name or "N/A"}
Tool Call ID: {self.tool_call_id or "N/A"}
Content:
{self.content}
"""


class GraphState(BaseModel):
    """State of the graph"""

    agents: dict[str, "AgentState"]


class AgentState(BaseModel):
    """State of an agent"""

    round: int = 0
    # Local environment for the agent
    local_env: LocalEnv
    # List of toolsets available to the agent
    toolsets: list[str] = ["noop"]
    # List of messages sent to the agent
    data_msgs: list[Message] = []
