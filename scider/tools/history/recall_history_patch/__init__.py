"""RecallHistoryTool — view the full message history including compacted summaries."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext


class RecallHistoryInput(BaseModel):
    last_n: int = Field(
        default=10,
        description="Number of recent messages to recall (default 10, max 50).",
    )


class RecallHistoryTool(BaseTool):
    name = "RecallHistory"
    description = (
        "Recall recent conversation history messages. "
        "Useful for reviewing what has been discussed or done."
    )
    input_schema = RecallHistoryInput
    _always_read_only = True

    def call(self, context: ToolContext, *, last_n: int = 10) -> str:
        agent_state = context.agent_state
        if agent_state is None:
            return "Error: no agent state available"

        last_n = min(max(last_n, 1), 50)
        messages = agent_state.messages[-last_n:]

        if not messages:
            return "No messages in history."

        parts = []
        for i, msg in enumerate(messages):
            parts.append(f"--- Message {i} Begin ---")
            parts.append(msg.to_plain_text(verbose_tool=False))
            parts.append(f"--- Message {i} End ---")

        return "\n".join(parts)
