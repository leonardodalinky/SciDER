"""
Context compression pipeline for the agent loop.

Runs before each LLM call in query() to keep context within budget.
Inspired by Claude Code's 5-level pipeline (§3 context-engineering),
we implement 3 levels suited to SciDER's synchronous architecture:

  Level 1: Tool Result Budget  — truncate oversized tool results in-place
  Level 2: History Snip        — clear old tool results the model no longer needs
  Level 3: Autocompact         — LLM-based full conversation summarization

Each level is progressively more expensive; the pipeline short-circuits
once context is under budget.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from .llms import ModelRegistry
from .types import HistoryState, Message
from .utils import parse_markdown_from_llm_response

# Lazy import to avoid circular dependency at module load time.
# PROMPTS is initialized via prompts.init() at startup.
_prompts = None


def _get_prompts():
    global _prompts
    if _prompts is None:
        from ..prompts import PROMPTS

        _prompts = PROMPTS
    return _prompts


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum characters per individual tool result before persistence to disk (Level 1)
TOOL_RESULT_MAX_CHARS = int(os.getenv("COMPACT_TOOL_RESULT_MAX_CHARS", 50_000))
# Preview size in characters for the reference message
TOOL_RESULT_PREVIEW_CHARS = int(os.getenv("COMPACT_TOOL_RESULT_PREVIEW_CHARS", 2_000))
# Default directory for persisted tool results — uses system temp dir.
# Can be overridden via env var or per-session via set_tool_results_dir().
_DEFAULT_TOOL_RESULTS_DIR: str | None = os.getenv("COMPACT_TOOL_RESULTS_DIR")
# Session-level override (set by query() from workspace path)
_tool_results_dir_override: str | None = None


def set_tool_results_dir(path: str | None) -> None:
    """Override the tool results directory for the current session."""
    global _tool_results_dir_override
    _tool_results_dir_override = path


def _get_tool_results_dir() -> str:
    if _tool_results_dir_override:
        return _tool_results_dir_override
    if _DEFAULT_TOOL_RESULTS_DIR:
        return _DEFAULT_TOOL_RESULTS_DIR
    import tempfile

    return os.path.join(tempfile.gettempdir(), "scider-tool-results")


# Level 2: keep the N most recent tool results intact; snip older ones
SNIP_KEEP_RECENT_TOOL_RESULTS = int(os.getenv("COMPACT_SNIP_KEEP_RECENT", 8))
SNIP_PREVIEW_CHARS = 200  # keep first N chars as summary when snipping

# Level 3: trigger autocompact when total tokens exceed this ratio of the threshold
AUTOCOMPACT_TOKEN_THRESHOLD = int(os.getenv("COMPACT_AUTOCOMPACT_TOKEN_THRESHOLD", 128_000))
AUTOCOMPACT_KEEP_RATIO = float(os.getenv("COMPACT_AUTOCOMPACT_KEEP_RATIO", 0.4))
AUTOCOMPACT_KEEP_FIRST_N = int(os.getenv("COMPACT_AUTOCOMPACT_KEEP_FIRST_N", 4))
# LLM model name for autocompact summarization (must be registered in ModelRegistry)
AUTOCOMPACT_MODEL = os.getenv("COMPACT_AUTOCOMPACT_MODEL", "history")

MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3


@contextmanager
def override_compact_settings(**overrides):
    """Temporarily override compact module-level settings.

    Usage::

        with override_compact_settings(AUTOCOMPACT_TOKEN_THRESHOLD=768_000):
            query(...)  # runs with higher threshold

    Supported keys: any module-level ALL_CAPS constant defined above
    (e.g. ``AUTOCOMPACT_TOKEN_THRESHOLD``, ``SNIP_KEEP_RECENT_TOOL_RESULTS``,
    ``AUTOCOMPACT_KEEP_RATIO``, etc.).
    """
    import scider.core.compact as _mod

    saved = {}
    for key, value in overrides.items():
        if not hasattr(_mod, key):
            raise AttributeError(f"compact module has no setting '{key}'")
        saved[key] = getattr(_mod, key)
        setattr(_mod, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(_mod, key, value)


@dataclass
class CompactState:
    """Tracks compression state across turns within a single query() call."""

    consecutive_autocompact_failures: int = 0
    total_tokens_freed_by_snip: int = 0
    # Tool result IDs already persisted (avoid re-writing on subsequent turns)
    persisted_tool_ids: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Level 1: Tool Result Budget
# ---------------------------------------------------------------------------


def _ensure_tool_results_dir() -> Path:
    """Ensure the tool results directory exists and return its path."""
    d = Path(_get_tool_results_dir())
    d.mkdir(parents=True, exist_ok=True)
    return d


def _persist_tool_result(content: str, tool_call_id: str | None) -> tuple[str, str]:
    """Persist tool result to disk and return (filepath, preview).

    Returns:
        (filepath, preview_text) — the path where content was saved, and a preview.
    """
    results_dir = _ensure_tool_results_dir()
    file_id = tool_call_id or uuid.uuid4().hex[:12]
    filepath = results_dir / f"{file_id}.txt"

    # Write only if not already persisted (idempotent)
    if not filepath.exists():
        filepath.write_text(content, encoding="utf-8")

    preview = content[:TOOL_RESULT_PREVIEW_CHARS]
    return str(filepath), preview


def _build_persisted_reference(filepath: str, original_size: int, preview: str) -> str:
    """Build the reference message that replaces the full tool result."""
    has_more = original_size > TOOL_RESULT_PREVIEW_CHARS
    msg = "<persisted-output>\n"
    msg += f"Output too large ({original_size:,} chars). Only the first {TOOL_RESULT_PREVIEW_CHARS:,} chars are shown below.\n"
    msg += "Work with this preview. Do NOT try to fetch the full output via WebFetch.\n\n"
    msg += preview
    if has_more:
        msg += "\n..."
    msg += "\n</persisted-output>"
    return msg


def _get_tool_max_result_size(tool_name: str | None) -> float:
    """Get per-tool max result size. Falls back to global default."""
    if tool_name:
        from ..tools.registry import ToolRegistry

        base_tools: dict = getattr(ToolRegistry.instance(), "_base_tools", {})
        if tool_name in base_tools:
            return base_tools[tool_name].max_result_size_chars
    return TOOL_RESULT_MAX_CHARS


def apply_tool_result_budget(history: list[Message], compact_state: CompactState) -> list[Message]:
    """Level 1: Persist oversized tool results to disk.

    When a tool result exceeds the tool's max_result_size_chars, the full
    content is saved to disk and replaced with a reference. Tools with
    max_result_size_chars=inf (e.g. read_file) are skipped to avoid
    circular reads.
    """
    for msg in history:
        if msg.role != "tool" or not msg.content:
            continue
        threshold = _get_tool_max_result_size(msg.tool_name)
        if threshold == float("inf") or len(msg.content) <= threshold:
            continue
        # Already persisted on a previous turn
        tool_id = msg.tool_call_id or id(msg)
        if tool_id in compact_state.persisted_tool_ids:
            continue

        original_len = len(msg.content)
        filepath, preview = _persist_tool_result(msg.content, msg.tool_call_id)
        msg.persisted_content_path = filepath
        msg.content = _build_persisted_reference(filepath, original_len, preview)
        msg._n_tokens = None  # invalidate cached token count

        compact_state.persisted_tool_ids.add(tool_id)
        logger.debug(
            "Level 1: Persisted tool result '{}' ({} chars) to {}",
            msg.tool_name or "unknown",
            original_len,
            filepath,
        )
    return history


# ---------------------------------------------------------------------------
# Level 2: History Snip
# ---------------------------------------------------------------------------


def apply_history_snip(history: list[Message]) -> tuple[list[Message], int]:
    """Level 2: Clear old tool results, keeping only the N most recent.

    Replaces content of old tool-role messages with a short placeholder.
    Returns (modified history, estimated tokens freed).
    """
    # Collect indices of tool-role messages (in reverse order = most recent first)
    tool_indices = [i for i, msg in enumerate(history) if msg.role == "tool"]

    if len(tool_indices) <= SNIP_KEEP_RECENT_TOOL_RESULTS:
        return history, 0

    # Indices to snip: all except the N most recent
    indices_to_snip = tool_indices[:-SNIP_KEEP_RECENT_TOOL_RESULTS]

    tokens_freed = 0
    for idx in indices_to_snip:
        msg = history[idx]
        if msg.content and not msg.content.startswith("[Snipped]"):
            old_tokens = msg.n_tokens
            msg.content_before_snip = msg.content
            # Keep a short preview so the compression LLM can still extract
            # key outcomes (e.g. "Output written on paper.pdf (5 pages)").
            preview = msg.content[:SNIP_PREVIEW_CHARS].rstrip()
            if len(msg.content) > SNIP_PREVIEW_CHARS:
                msg.content = f"[Snipped] {preview} ..."
            # else: content is short enough, no need to snip
            else:
                continue
            msg._n_tokens = None  # invalidate cache
            new_tokens = msg.n_tokens
            tokens_freed += max(old_tokens - new_tokens, 0)

    if tokens_freed > 0:
        logger.debug(
            "Level 2: Snipped {} old tool results, freed ~{} tokens",
            len(indices_to_snip),
            tokens_freed,
        )

    return history, tokens_freed


# ---------------------------------------------------------------------------
# Skill reminder (re-inject invoked skills after compaction)
# ---------------------------------------------------------------------------

# Token budget for re-injected skills. Each skill is truncated to this limit;
# if the total exceeds the budget, least-recently-added skills are dropped.
_SKILL_REMINDER_MAX_PER_SKILL = 5_000  # chars (~1.5k tokens)
_SKILL_REMINDER_TOTAL_BUDGET = 25_000  # chars (~7.5k tokens)


def _build_skill_reminder(agent_state: HistoryState) -> Message | None:
    """Build a meta user message containing invoked skill contents.

    Returns None if no skills have been invoked.
    """
    if not agent_state.invoked_skills:
        return None

    # Inject the first skill (often the orchestrator / master roadmap) and
    # the most recently loaded skill (the one the agent is currently working
    # with). This ensures the agent retains high-level pipeline awareness
    # while staying focused on the current step.
    items = list(agent_state.invoked_skills.items())

    def _truncate(content: str) -> str:
        if len(content) <= _SKILL_REMINDER_MAX_PER_SKILL:
            return content
        return content[:_SKILL_REMINDER_MAX_PER_SKILL] + (
            "\n\n... (truncated — load the skill again for the full content)"
        )

    parts: list[str] = []
    first_name, first_content = items[0]
    parts.append(_truncate(first_content))

    if len(items) > 1:
        last_name, last_content = items[-1]
        parts.append(_truncate(last_content))

    text = (
        "<system-reminder>\n"
        "The following skills were loaded earlier in this session. "
        "Continue to follow their guidelines:\n\n"
        + "\n\n---\n\n".join(parts)
        + "\n</system-reminder>"
    )

    return Message(
        role="user",
        content=text,
        agent_sender="compact",
        is_meta=True,
    )


# ---------------------------------------------------------------------------
# Level 3: Autocompact (LLM-based summarization)
# ---------------------------------------------------------------------------


def should_autocompact(
    agent_state: HistoryState,
    compact_state: CompactState,
    snip_tokens_freed: int = 0,
) -> bool:
    """Check whether autocompact should run."""
    if compact_state.consecutive_autocompact_failures >= MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES:
        logger.warning(
            "Autocompact circuit breaker: {} consecutive failures, skipping",
            compact_state.consecutive_autocompact_failures,
        )
        return False

    effective_tokens = agent_state.total_tokens - snip_tokens_freed
    return effective_tokens >= AUTOCOMPACT_TOKEN_THRESHOLD


def _split_compress_keep(
    current_messages: list[Message],
    total_tokens: int,
) -> tuple[list[Message], list[Message]]:
    """Split messages into (to_compress, to_keep).

    Keeps first N messages and the tail intact; compresses the middle.
    Respects tool call / tool result boundaries.
    """
    keep_first_n = AUTOCOMPACT_KEEP_FIRST_N

    # Filter out boundary markers for splitting logic
    non_boundary = [m for m in current_messages if not m.is_compact_boundary]

    # Adjust keep_first_n to avoid splitting tool call / tool result pairs
    if keep_first_n > 0 and keep_first_n < len(non_boundary):
        last_kept = non_boundary[keep_first_n - 1]
        if last_kept.role == "assistant" and last_kept.tool_calls:
            keep_first_n -= 1
        elif last_kept.role == "tool":
            for i in range(keep_first_n - 2, -1, -1):
                if non_boundary[i].role == "assistant":
                    keep_first_n = i
                    break

    compressible = non_boundary[keep_first_n:]
    if not compressible:
        return [], non_boundary

    # Select messages to compress up to the token budget
    tokens_to_compress = (1 - AUTOCOMPACT_KEEP_RATIO) * total_tokens
    n_tokens = 0
    count = 0
    for msg in compressible:
        if n_tokens > tokens_to_compress:
            break
        count += 1
        n_tokens += msg.n_tokens

    # Ensure we don't split a tool call / tool result boundary
    if count > 0:
        last_msg = compressible[count - 1]
        if last_msg.role == "assistant" and last_msg.tool_calls:
            for msg in compressible[count:]:
                if msg.role == "tool":
                    count += 1
                else:
                    break
        elif last_msg.role == "tool":
            for msg in compressible[count:]:
                if msg.role == "tool":
                    count += 1
                else:
                    break

    to_compress = compressible[:count]
    to_keep = non_boundary[:keep_first_n] + compressible[count:]
    return to_compress, to_keep


def apply_autocompact(
    agent_state: HistoryState,
    compact_state: CompactState,
) -> bool:
    """Level 3: LLM-based conversation summarization.

    Selects messages to compress, generates a summary via LLM, then
    replaces the entire history with: boundary + summary + kept messages.

    Returns True if compaction succeeded, False otherwise.
    """
    current_messages = agent_state.messages
    total_tokens = agent_state.total_tokens

    to_compress, to_keep = _split_compress_keep(current_messages, total_tokens)

    if len(to_compress) == 0:
        logger.debug("Autocompact: no messages to compress")
        return False

    # Format messages for LLM
    input_msgs_texts = []
    for i, msg in enumerate(to_compress):
        plain = msg.to_plain_text(verbose_tool=False)
        input_msgs_texts.append(f"--- Message {i} Begin ---\n{plain}\n--- Message {i} End ---")
    message_text = "\n".join(input_msgs_texts)

    prompts = _get_prompts()
    system_prompt = prompts.history.compression_system_prompt.render()
    user_prompt = prompts.history.compression_user_prompt.render(
        n_messages=len(to_compress),
        message_text=message_text,
    )

    try:
        compressed_msg = ModelRegistry.completion(
            AUTOCOMPACT_MODEL,
            [Message(role="user", content=user_prompt)],
            system_prompt=system_prompt,
            agent_sender="compact",
        )

        compressed_text = parse_markdown_from_llm_response(compressed_msg)

        summary_message = Message(
            role="assistant",
            content=f"# Conversation Summary\n\n"
            f"This summary covers {len(to_compress)} messages.\n\n"
            f"{compressed_text}",
            agent_sender="compact",
            is_meta=True,
        ).with_log()

        # Build the post-compact message list: summary + skill reminder + kept
        post_messages = [summary_message]

        # Re-inject invoked skills so the agent doesn't need to re-load them.
        skill_reminder = _build_skill_reminder(agent_state)
        if skill_reminder is not None:
            post_messages.append(skill_reminder)

        post_messages.extend(to_keep)

        # Complete array replacement: boundary + post_messages
        agent_state.compact(
            summary_messages=post_messages,
            trigger="auto",
        )

        compact_state.consecutive_autocompact_failures = 0
        logger.info(
            "Autocompact: compressed {} messages, kept {} messages{}",
            len(to_compress),
            len(to_keep),
            (
                f", re-injected {len(agent_state.invoked_skills)} skill(s)"
                if agent_state.invoked_skills
                else ""
            ),
        )
        return True

    except Exception as e:
        compact_state.consecutive_autocompact_failures += 1
        logger.error(
            "Autocompact failed (attempt {}): {}",
            compact_state.consecutive_autocompact_failures,
            e,
        )
        return False


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_compression_pipeline(
    agent_state: HistoryState,
    compact_state: CompactState,
) -> None:
    """Run the 3-level compression pipeline on the agent state.

    Called before each LLM call in query(). Mutates agent_state in-place.
    """
    # Level 1: Tool Result Budget — persist oversized results to disk
    apply_tool_result_budget(agent_state.messages, compact_state)

    # Level 2: History Snip — clear old tool results
    _, snip_tokens_freed = apply_history_snip(agent_state.messages)
    compact_state.total_tokens_freed_by_snip += snip_tokens_freed

    # Level 3: Autocompact — expensive LLM call, only when needed
    if should_autocompact(agent_state, compact_state, snip_tokens_freed):
        logger.info(
            "Autocompact triggered: {} tokens (threshold {})",
            agent_state.total_tokens,
            AUTOCOMPACT_TOKEN_THRESHOLD,
        )
        apply_autocompact(agent_state, compact_state)
