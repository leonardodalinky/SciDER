from __future__ import annotations

from pathlib import Path
from typing import Self

import tiktoken
from functional import seq
from langgraph.graph import START
from litellm import Message as LLMessage
from pydantic import BaseModel, field_validator
from rich.console import Console
from rich.style import Style

from .exec import SessionBase

console = Console()

styles = {
    "assistant": Style(color="green"),
    "user": Style(color="white"),
    "system": Style(color="red"),
    "tool": Style(color="magenta"),
    "function": Style(color="yellow"),
}

ENCODING = tiktoken.get_encoding("cl100k_base")


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
        "tool_name",
        "completion_tokens",
        "prompt_tokens",
        "_n_tokens",
    ]

    # --- tool call fields ---
    tool_call_id: str | None = None

    # --- custom fields ---
    llm_sender: str | None = None
    agent_sender: str | None = None
    tool_name: str | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    _n_tokens: int | None = None

    @classmethod
    def from_ll_message(cls, msg: LLMessage) -> "Message":
        o = cls(**msg.__dict__)
        return o

    @property
    def n_tokens(self) -> int:
        """
        Returns the number of tokens in the message.
        """
        if self._n_tokens is not None:
            return self._n_tokens
        if self.prompt_tokens is not None:
            return self.prompt_tokens
        # Calculate n_tokens if not cached
        self._n_tokens = len(ENCODING.encode(self.to_plain_text()))
        return self._n_tokens

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

    def to_ll_message(self, exclude_none: bool = True) -> LLMessage | dict:
        return LLMessage(
            **self.model_dump(exclude=self.__CUSTOM_FIELDS__, exclude_none=exclude_none)  # type: ignore
        )

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
    toolsets: list[str] = []


class HistoryState(BaseModel):
    # List of messages sent to the agent
    history: list[Message] = []
    # List of patches to the history, used to compress the history
    # NOTE: patches are applied in order by patch_id, and patched history could still be patched in the next patches.
    history_patches: list["HistoryState.HistoryPatch"] = []
    node_history: list[str] = [START]

    class HistoryPatch(BaseModel):
        # patch id, used to identify the patch
        patch_id: int
        # start index (inclusive)
        start_idx: int
        # end index (exclusive)
        end_idx: int
        # The compressed/summarized message that replaces the range
        patched_message: Message

        @property
        def n_messages(self) -> int:
            return max(self.end_idx - self.start_idx, 0)

    @property
    def patched_history(self) -> list[Message]:
        """
        Returns the history with all patches applied in order.
        Each patch replaces a range of messages with a single compressed message.
        """
        if not self.history_patches:
            return self.history.copy()

        # Apply patches in order
        result = self.history.copy()
        # Sort patches by patch_id to ensure proper order
        sorted_patches = sorted(self.history_patches, key=lambda p: p.patch_id)

        for patch in sorted_patches:
            # Adjust indices based on previous patches
            adjusted_start = patch.start_idx
            adjusted_end = patch.end_idx

            # Validate indices
            if adjusted_start < 0 or adjusted_end > len(result):
                continue

            # Replace the range with the patched message
            result = result[:adjusted_start] + [patch.patched_message] + result[adjusted_end:]

        return result

    @property
    def total_patched_tokens(self) -> int:
        return sum(m.n_tokens for m in self.patched_history)

    @property
    def round(self) -> int:
        return len(self.node_history) - 1

    def add_node_history(self, node_name: str) -> None:
        self.node_history.append(node_name)

    def next_patch_id(self) -> int:
        if not self.history_patches or len(self.history_patches) == 0:
            return 0
        return max(p.patch_id for p in self.history_patches) + 1

    def partial_history_of_patch(self, patch_id: int) -> list[Message]:
        """Get the history of a specific patch."""
        patch = self.get_patch_by_id(patch_id)
        if patch is None:
            raise ValueError(f"Patch {patch_id} not found")

        # sort patches by patch_id to ensure proper order
        sorted_patches = sorted(self.history_patches, key=lambda p: p.patch_id)

        # apply patches in order
        his = self.history.copy()
        for patch in sorted_patches:
            if patch.patch_id >= patch_id:
                break
            his = his[: patch.start_idx] + [patch.patched_message] + his[patch.end_idx :]

        # get the history of the patch
        return his[patch.start_idx : patch.end_idx]

    def add_message(self, message: Message) -> None:
        """Add a new message to the history."""
        self.history.append(message)

    def get_patch_by_id(self, patch_id: int) -> HistoryState.HistoryPatch | None:
        """
        Get a patch by its ID.

        Args:
            patch_id: The ID of the patch to retrieve

        Returns:
            The patch if found, None otherwise
        """
        return (
            seq(self.history_patches)
            .filter(lambda patch: patch.patch_id == patch_id)
            .head_option(no_wrap=True)
        )  # type: ignore


class RBankState(BaseModel):
    # session dir (short-term mem storage)
    sess_dir: str | Path
    # long-term mem save dirs (input & output)
    long_term_mem_dir: str | Path
    # project mem save dirs (input & output)
    project_mem_dir: str | Path


class ExecState(BaseModel):
    # Current execution state
    session: SessionBase

    @field_validator("session")
    @classmethod
    def validate_session_is_subclass(cls, v):
        if type(v) is SessionBase:
            raise ValueError(
                "session must be an instance of a SessionBase subclass, not SessionBase itself"
            )
        return v
