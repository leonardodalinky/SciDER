from typing import Self

from litellm import Message as LLMessage
from pydantic import BaseModel
from rich.console import Console
from rich.style import Style

console = Console()

styles = {
    "assistant": Style(color="green"),
    "user": Style(color="white"),
    "system": Style(color="red"),
    "tool": Style(color="magenta"),
    "function": Style(color="yellow"),
}


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

    @property
    def reasoning_text(self) -> str | None:
        if not hasattr(self, "reasoning_content"):
            return None
        elif self.reasoning_content is None:
            return None
        return self.reasoning_content

    @reasoning_text.setter
    def reasoning_text(self, value: str | None):
        self.reasoning_content = value

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

    def to_plain_text(self, verbose_tool: bool = False) -> str:
        # format .tool_calls
        tool_text = ""
        if self.tool_name:
            tool_text += f"- Tool Name: {self.tool_name}\n"
        if verbose_tool and self.tool_call_id:
            tool_text += f"- Tool Call ID: {self.tool_call_id}\n"
        if self.tool_calls and len(self.tool_calls) > 0:
            tool_text += "- Tool Calls:\n"
            for tool_call in self.tool_calls:
                if verbose_tool:
                    tool_text += f"  - {tool_call.function.name}({tool_call.function.arguments}), id: {tool_call.id}\n"
                else:
                    tool_text += f"  - {tool_call.function.name}\n"

        if self.reasoning_text is None:
            return f"""\
## Metadata

- Role: {self.role}
- Agent Sender: {self.agent_sender or "N/A"}
{tool_text}

## Content

{self.content}
"""
        else:
            return f"""\
## Metadata

- Role: {self.role}
- Agent Sender: {self.agent_sender or "N/A"}
{tool_text}

## Thinking Process

{self.reasoning_text}

## Content

{self.content}
"""

    def with_log(self, cond: bool | None = None) -> Self:
        """
        Log the message to console and other loggers. Returns self.

        Returns:
            self
        """
        if cond is not None and not cond:
            return self

        if self.agent_sender:
            text = f"""
--- Message from `{self.role}` of Agent `{self.agent_sender}`  ---
{self.to_plain_text(verbose_tool=True)}
--- Message End ---
"""
        else:
            text = f"""
--- Message from `{self.role}`  ---
{self.to_plain_text(verbose_tool=True)}
--- Message End ---
"""
        console.print(text, style=styles.get(self.role, Style()))

        return self


class ToolsetState(BaseModel):
    # List of toolsets available to the agent
    toolsets: list[str] = ["todo"]
