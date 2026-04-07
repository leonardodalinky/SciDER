"""Conversation history export utility.

Saves agent conversation history to workspace as JSON for debugging and review.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from scider.core.types import Message


def save_conversation_history(
    history: list[Message] | list,
    path: Path,
    agent_name: str = "agent",
) -> Path:
    """Save conversation history to a JSON file.

    Args:
        history: List of Message objects or dicts.
        path: Output file path (e.g., workspace/data_agent_history.json).
        agent_name: Agent name for logging.

    Returns:
        The path where the history was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for msg in history:
        if isinstance(msg, Message):
            record = {
                "role": msg.role,
                "content": msg.content,
                "agent_sender": msg.agent_sender,
                "tool_name": msg.tool_name,
                "tool_call_id": msg.tool_call_id,
                "is_meta": msg.is_meta,
            }
            if msg.tool_calls:
                record["tool_calls"] = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name if tc.function else None,
                            "arguments": tc.function.arguments if tc.function else None,
                        },
                    }
                    for tc in msg.tool_calls
                ]
        elif isinstance(msg, dict):
            record = msg
        else:
            record = {"content": str(msg)}
        records.append(record)

    path.write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str))
    logger.info("Saved {} messages to {}", len(records), path)
    return path
