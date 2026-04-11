"""Conversation history export utility.

Saves agent conversation history to workspace as JSON for debugging and review.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Iterator

from loguru import logger

from scider.core.types import Message, message_listener


@contextlib.contextmanager
def capture_messages() -> Iterator[list[Message]]:
    """Context manager that captures every Message added to any HistoryState.

    Returns a list that the caller can read after the context exits. The list
    is populated in real time, so even if the agent crashes mid-run, the
    captured messages so far are preserved.

    Usage::

        with capture_messages() as captured:
            graph.invoke(state)  # may raise
        # captured is the message history regardless of success/failure
    """
    captured: list[Message] = []

    def _listener(msg: Message) -> None:
        captured.append(msg)

    with message_listener(_listener):
        yield captured


def _restore_original_content(msg: Message) -> str | None:
    """Get the original pre-compression content for a message.

    - Level 2 snip: ``content_before_snip`` holds the original.
    - Level 1 persist: the full content lives at ``persisted_content_path``.
    - Otherwise: current ``content``.
    """
    if msg.content_before_snip is not None:
        return msg.content_before_snip
    if msg.persisted_content_path:
        try:
            return Path(msg.persisted_content_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(
                "Failed to restore persisted content from {}: {}", msg.persisted_content_path, e
            )
    return msg.content


def save_conversation_history(
    history: list[Message] | list,
    path: Path,
    agent_name: str = "agent",
) -> Path:
    """Save conversation history to a JSON file.

    The saved history preserves:
    - Compact boundary markers (so you can see where compaction happened)
    - Original tool result content (pre-Level-1 persist and pre-Level-2 snip)

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
                "content": _restore_original_content(msg),
                "agent_sender": msg.agent_sender,
                "tool_name": msg.tool_name,
                "tool_call_id": msg.tool_call_id,
                "is_meta": msg.is_meta,
            }
            # Preserve compact boundary markers in the saved log.
            if msg.is_compact_boundary:
                record["is_compact_boundary"] = True
                if msg.compact_metadata:
                    record["compact_metadata"] = msg.compact_metadata
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


def save_subagent_history(
    history: list,
    workspace_path: Path | str,
    subagent_type: str,
) -> Path | None:
    """Save a subagent's conversation history under <workspace>/subagents/.

    Files are named ``<subagent_type>_<NNN>.json`` with NNN auto-incrementing
    so multiple invocations of the same subagent type don't overwrite each other.

    Args:
        history: List of Message objects from the subagent state.
        workspace_path: Parent agent's workspace directory.
        subagent_type: Name of the subagent (e.g. "coding", "paper_search", "critic").

    Returns:
        The path where the history was saved, or None if history was empty/save failed.
    """
    if not history:
        return None

    try:
        subagents_dir = Path(workspace_path) / "subagents"
        subagents_dir.mkdir(parents=True, exist_ok=True)

        # Find next available index for this subagent type
        existing = list(subagents_dir.glob(f"{subagent_type}_*.json"))
        next_idx = len(existing) + 1
        path = subagents_dir / f"{subagent_type}_{next_idx:03d}.json"

        return save_conversation_history(history, path, agent_name=subagent_type)
    except Exception as e:
        logger.warning("Failed to save {} subagent history: {}", subagent_type, e)
        return None
