"""Lightweight file-based memory system.

Provides cross-session memory persistence via markdown files in .scider/memory/.
Modeled after Claude Code's memory system (Chapter 8) but without semantic recall.

Memory types (closed taxonomy):
- user: User identity, preferences, knowledge background
- feedback: Behavioral corrections and confirmations
- project: Project dynamics, decisions, deadlines
- reference: Pointers to external systems

Loading:
- MEMORY.md index is injected into the system prompt every session
- Agent can Read individual memory files for details
- Agent can FileWrite new memories and update MEMORY.md

No LLM-based semantic recall — just index injection + on-demand file reads.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

# Limits matching Claude Code
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000

# Default memory directory name under .scider/
_MEMORY_SUBDIR = "memory"


def _is_memory_read_enabled() -> bool:
    """Check if memory reading is enabled. Default: True."""
    return os.getenv("SCIDER_MEMORY_READ", "true").strip().lower() in {"1", "true", "yes", "y"}


def _is_memory_write_enabled() -> bool:
    """Check if memory writing is enabled. Default: True."""
    return os.getenv("SCIDER_MEMORY_WRITE", "true").strip().lower() in {"1", "true", "yes", "y"}


_MEMORY_READ_GUIDANCE = """\
You have a persistent memory system at the path shown above. \
MEMORY.md is the index — each entry is a one-line link to a memory file.

## Reading memories
- The index is already loaded above. To read a specific memory, use Read on the linked file.
- Before acting on a memory, verify it's still current (files may have changed since it was written).
"""

_MEMORY_WRITE_GUIDANCE = """\
## Writing memories
When you learn something worth remembering across sessions, save it:
1. Write the memory file with FileWrite to `.scider/memory/filename.md` using this format:
   ```
   ---
   name: short name
   description: one-line description (used for index)
   type: user | feedback | project | reference
   ---
   Memory content here.
   For feedback/project types, include **Why:** and **How to apply:** lines.
   ```
2. Update MEMORY.md index: add a one-line entry `- [Name](filename.md) — description`

## What to save
- user: User role, preferences, domain knowledge
- feedback: When the user corrects you ("don't do X") OR confirms a good approach
- project: Decisions, deadlines, who's doing what (convert relative dates to absolute)
- reference: Where to find info in external systems

## What NOT to save
- Anything derivable from code, git history, or existing files
- Temporary task state or current conversation context
- Content already in skills or CLAUDE.md
"""


def _build_guidance() -> str:
    """Build memory guidance based on enabled features."""
    parts = [_MEMORY_READ_GUIDANCE]
    if _is_memory_write_enabled():
        parts.append(_MEMORY_WRITE_GUIDANCE)
    else:
        parts.append("Memory writing is disabled. You can only read existing memories.\n")
    return "\n".join(parts)


def load_memory_index() -> str:
    """Load the MEMORY.md index by walking up from cwd to root, plus home.

    Checks .scider/memory/MEMORY.md in each directory. Returns the FIRST
    one found (closest to workspace wins for memory, since memory is
    per-project).
    """
    from .scider_context import walk_up_dirs

    # Walk up but check in REVERSE (workspace-first) so closest wins
    dirs = list(reversed(walk_up_dirs()))
    for d in dirs:
        memory_dir = d / ".scider" / _MEMORY_SUBDIR
        index_path = memory_dir / "MEMORY.md"
        if index_path.is_file():
            return _read_and_truncate(index_path, str(memory_dir))
    return ""


def _read_and_truncate(path: Path, memory_dir: str) -> str:
    """Read MEMORY.md and truncate if needed."""
    try:
        content = path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning("Failed to read {}: {}", path, e)
        return ""

    if not content:
        return ""

    lines = content.split("\n")
    truncated = False

    # Line truncation
    if len(lines) > MAX_INDEX_LINES:
        lines = lines[:MAX_INDEX_LINES]
        truncated = True

    result = "\n".join(lines)

    # Byte truncation
    if len(result.encode("utf-8")) > MAX_INDEX_BYTES:
        while len(result.encode("utf-8")) > MAX_INDEX_BYTES and "\n" in result:
            result = result[: result.rindex("\n")]
        truncated = True

    if truncated:
        result += (
            "\n\n> WARNING: MEMORY.md was truncated. Keep entries to one line under ~150 chars."
        )

    header = f"Memory directory: `{memory_dir}/`\n\n"
    return header + result


def build_memory_prompt_section() -> str:
    """Build the complete memory section for the system prompt.

    Returns empty string if memory reading is disabled or no index exists.
    Controlled by SCIDER_MEMORY_READ env var (default: true).
    Writing guidance is included only if SCIDER_MEMORY_WRITE is also true.
    """
    if not _is_memory_read_enabled():
        return ""

    index = load_memory_index()
    if not index:
        return ""

    return f"{index}\n\n{_build_guidance()}"


def ensure_memory_dir() -> Path:
    """Ensure the project-level memory directory exists. Returns the path."""
    d = Path(".scider") / _MEMORY_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d
