from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Callable, Iterator, Self

# Global listeners invoked on every HistoryState.add_message() call.
# Multiple subscribers are supported (e.g. UI layer + workflow history capture).
_message_listeners: list[Callable[["Message"], None]] = []


def add_message_listener(listener: Callable[["Message"], None]) -> None:
    """Register a listener that fires on every add_message() call.

    Multiple listeners can be registered. Each will be called in registration
    order. Exceptions from listeners are caught and ignored.
    """
    _message_listeners.append(listener)


def remove_message_listener(listener: Callable[["Message"], None]) -> None:
    """Unregister a previously-registered message listener."""
    try:
        _message_listeners.remove(listener)
    except ValueError:
        pass


@contextlib.contextmanager
def message_listener(
    listener: Callable[["Message"], None],
) -> Iterator[Callable[["Message"], None]]:
    """Context manager: register a listener for the duration of the block.

    Always unregisters on exit, even if the block raises.
    """
    add_message_listener(listener)
    try:
        yield listener
    finally:
        remove_message_listener(listener)


# --- Backwards-compat shim ---
# Old single-callback API. Kept for streamlit-client and other consumers that
# haven't migrated yet. Internally translates to add/remove listener calls.
_legacy_callback: Callable[["Message"], None] | None = None


def set_on_message_callback(callback: Callable[["Message"], None] | None) -> None:
    """[Deprecated] Set a single global callback. Prefer add_message_listener().

    Replacing the callback removes the previous one and registers the new one.
    Pass ``None`` to clear.
    """
    global _legacy_callback
    if _legacy_callback is not None:
        remove_message_listener(_legacy_callback)
        _legacy_callback = None
    if callback is not None:
        add_message_listener(callback)
        _legacy_callback = callback


import tiktoken
from langgraph.graph import START
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

ENCODING = tiktoken.get_encoding("cl100k_base")

# Compact boundary marker content
COMPACT_BOUNDARY_CONTENT = "[compact_boundary]"


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
        "is_meta",
        "is_compact_boundary",
        "compact_metadata",
        "persisted_content_path",
        "content_before_snip",
        "finish_reason",
        "tool_result_images",
        "tool_result_documents",
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
    # True for programmatically injected messages (recovery nudges, system context, etc.)
    # that are not real user/LLM interactions. Excluded when serializing to litellm.
    is_meta: bool = False
    # LLM finish reason: "stop", "length", "tool_calls", etc.
    finish_reason: str | None = None
    # Compact boundary marker — inserted by autocompact to delimit compacted regions
    is_compact_boundary: bool = False
    # Metadata about compaction (trigger type, pre-token count, messages summarized, etc.)
    compact_metadata: dict | None = None
    # Path to the persisted full content on disk (set by Level 1 tool result budget).
    # For debugging — not sent to LLM.
    persisted_content_path: str | None = None
    # Original content before Level 2 snip replaced it with a placeholder.
    # Kept in memory for debugging — not sent to LLM.
    content_before_snip: str | None = None
    # Images attached to a tool result (role=="tool"). Each entry is
    # ``{"media_type": "image/png", "data": "<pure base64>"}``. Excluded from
    # the raw LLMessage dict; injected into the LLM payload by the
    # provider-aware serializer in ``scider/core/llms.py``.
    tool_result_images: list[dict] | None = None
    # Documents (currently PDFs) attached to a tool result (role=="tool"). Each
    # entry is ``{"media_type": "application/pdf", "data": "<pure base64>",
    # "filename": "<original filename or None>"}``. Excluded from the raw
    # LLMessage dict; injected by the provider-aware serializer in
    # ``scider/core/llms.py``.
    tool_result_documents: list[dict] | None = None

    @classmethod
    def from_ll_message(cls, msg: LLMessage) -> "Message":
        o = cls(**msg.__dict__)
        return o

    @classmethod
    def make_compact_boundary(
        cls,
        *,
        trigger: str = "auto",
        pre_tokens: int = 0,
        messages_summarized: int = 0,
    ) -> "Message":
        """Create a compact boundary marker message."""
        return cls(
            role="system",
            content=COMPACT_BOUNDARY_CONTENT,
            is_compact_boundary=True,
            is_meta=True,
            compact_metadata={
                "trigger": trigger,
                "pre_tokens": pre_tokens,
                "messages_summarized": messages_summarized,
            },
        )

    @property
    def n_tokens(self) -> int:
        """Tokens contributed by THIS message alone.

        For assistant messages produced via the API, use ``completion_tokens``
        (the output size of this message). Do NOT use ``prompt_tokens`` —
        that is the cumulative input of the whole request that generated
        this message (system + full history), and summing it across messages
        triple-counts the history and fires autocompact far too early.
        """
        if self._n_tokens is not None:
            return self._n_tokens
        if self.role == "assistant" and self.completion_tokens is not None:
            return self.completion_tokens
        # Fallback: local tiktoken estimate of the message's plain-text form.
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

        from rich.markup import escape

        text = self.to_plain_text(verbose_tool=True)
        text = escape(text)

        if self.agent_sender:
            text = f"""
--- Message from `{self.role}` of Agent `{self.agent_sender}`  ---
{text}
--- Message End ---
"""
        else:
            text = f"""
--- Message from `{self.role}`  ---
{text}
--- Message End ---
"""
        console.print(text, style=styles.get(self.role, Style()))

        return self


class HistoryState(BaseModel):
    # Flat message array — the single source of truth.
    # Autocompact replaces messages in-place (boundary marker + summary).
    history: list[Message] = []
    node_history: list[str] = [START]

    # Skills invoked during this agent loop (name → content).
    # Persisted across autocompact so the agent doesn't need to re-load them.
    invoked_skills: dict[str, str] = {}

    @property
    def messages(self) -> list[Message]:
        """Return messages after the last compact boundary.

        This is the primary way to access the current conversation history.
        Messages before the boundary have been summarized and replaced.
        """
        return get_messages_after_boundary(self.history)

    @property
    def total_tokens(self) -> int:
        return sum(m.n_tokens for m in self.messages)

    @property
    def round(self) -> int:
        return len(self.node_history) - 1

    def add_node_history(self, node_name: str) -> None:
        self.node_history.append(node_name)

    def add_message(self, message: Message) -> None:
        """Add a new message to the history."""
        self.history.append(message)
        for listener in list(_message_listeners):
            try:
                listener(message)
            except Exception:
                pass

    def compact(self, summary_messages: list[Message], *, trigger: str = "auto") -> None:
        """Replace all messages before current boundary with a compacted summary.

        Inserts a compact boundary marker followed by the summary messages.
        Old messages are discarded from the live history but listeners
        (e.g. ``capture_messages``) still see the boundary + summary so the
        external log keeps a complete timeline.
        """
        current_messages = self.messages
        pre_tokens = sum(m.n_tokens for m in current_messages)

        boundary = Message.make_compact_boundary(
            trigger=trigger,
            pre_tokens=pre_tokens,
            messages_summarized=len(current_messages),
        )

        # Complete replacement: boundary + summary + nothing else
        # (caller is responsible for keeping any messages that should survive)
        self.history = [boundary] + summary_messages

        # Notify listeners of the new boundary + summary messages so external
        # captures (history_export.capture_messages) can log the compaction.
        for emitted in [boundary, *summary_messages]:
            for listener in list(_message_listeners):
                try:
                    listener(emitted)
                except Exception:
                    pass

    def normalize_for_api(self) -> list[Message]:
        """Normalize message history for LLM API consumption.

        Ensures:
        1. No compact boundary markers sent to API
        2. Every tool_use (assistant with tool_calls) has matching tool results
        3. Every tool result has a preceding tool_use
        4. No system messages in the middle (only as first message)
        """
        msgs = self.messages
        result: list[Message] = []

        for msg in msgs:
            # Skip compact boundary markers
            if msg.is_compact_boundary:
                continue
            result.append(msg)

        # Ensure tool call / tool result pairing
        result = _ensure_tool_pairing(result)

        return result

    # --- Backward compatibility ---

    @property
    def patched_history(self) -> list[Message]:
        """Backward compatibility alias for messages."""
        return self.messages

    @property
    def total_patched_tokens(self) -> int:
        """Backward compatibility alias for total_tokens."""
        return self.total_tokens


def get_messages_after_boundary(messages: list[Message]) -> list[Message]:
    """Return messages after the last compact boundary marker.

    If no boundary exists, returns all messages.
    The boundary marker itself is included in the result.
    """
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].is_compact_boundary:
            return messages[i:]
    return messages


def _ensure_tool_pairing(messages: list[Message]) -> list[Message]:
    """Ensure every tool_use has a matching tool_result and vice versa.

    Fixes orphaned tool calls/results that can occur after compaction
    or session crashes.
    """
    result = []
    # Collect tool_call IDs from assistant messages that have tool_calls
    pending_tool_calls: set[str] = set()

    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.id:
                    pending_tool_calls.add(tc.id)
            result.append(msg)
        elif msg.role == "tool" and msg.tool_call_id:
            if msg.tool_call_id in pending_tool_calls:
                pending_tool_calls.discard(msg.tool_call_id)
                result.append(msg)
            else:
                # Orphaned tool result — skip it
                pass
        else:
            result.append(msg)

    # For any pending tool calls without results, inject synthetic results
    if pending_tool_calls:
        # Find the assistant message that has these tool calls and inject after it
        for i, msg in enumerate(result):
            if msg.role == "assistant" and msg.tool_calls:
                orphan_ids = {tc.id for tc in msg.tool_calls if tc.id in pending_tool_calls}
                if orphan_ids:
                    insert_pos = i + 1
                    # Skip past any existing tool results for this message
                    while insert_pos < len(result) and result[insert_pos].role == "tool":
                        insert_pos += 1
                    for oid in orphan_ids:
                        tc = next(tc for tc in msg.tool_calls if tc.id == oid)
                        result.insert(
                            insert_pos,
                            Message(
                                role="tool",
                                tool_call_id=oid,
                                tool_name=tc.function.name if tc.function else None,
                                content="[Tool result lost due to compaction]",
                                is_meta=True,
                            ),
                        )
                        insert_pos += 1
                        pending_tool_calls.discard(oid)

    return result
