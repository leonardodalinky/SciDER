"""SCIDER project context discovery and loading.

Provides the walk-up directory discovery used by SCIDER.md, skills, and memory.
Mirrors Claude Code's CLAUDE.md approach:

- Walk from workspace upward to filesystem root
- At each directory, check for .scider/ content
- Append home directory (~/.scider/) if not already visited
- All found content is concatenated root-first (workspace-last = highest priority)
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

MAX_SCIDER_MD_CHARS = 40_000

# Filenames to check in each directory (order matters within a dir)
_CANDIDATES = [
    ".scider/SCIDER.md",
    "SCIDER.md",
]

_INSTRUCTION_HEADER = (
    "Project instructions from SCIDER.md files are shown below. "
    "Follow these instructions when working in this project."
)

# Per-workspace cache to avoid re-reading files on every query() call.
_cache: dict[str, str | None] = {}


# ---------------------------------------------------------------------------
# Shared walk-up utility
# ---------------------------------------------------------------------------


def walk_up_dirs(start: str | Path | None = None) -> list[Path]:
    """Build a root-first directory chain from ``start`` up to the filesystem root.

    Also appends ``~/.scider``'s parent (home dir) if it wasn't already visited
    during the walk.

    The returned list is ordered root → start, so that closer directories appear
    later (higher effective priority in LLM context).
    """
    base = Path(start).resolve() if start else Path.cwd().resolve()

    # Collect start → root
    dirs: list[Path] = []
    seen: set[str] = set()
    current = base
    while current != current.parent:
        resolved = str(current)
        if resolved not in seen:
            dirs.append(current)
            seen.add(resolved)
        current = current.parent

    # Reverse to root-first
    dirs.reverse()

    # Append home directory if not already visited
    home = Path.home().resolve()
    if str(home) not in seen:
        dirs.append(home)

    return dirs


# ---------------------------------------------------------------------------
# SCIDER.md loader
# ---------------------------------------------------------------------------


def load_scider_md(workspace_path: str | Path | None = None) -> str | None:
    """Discover and load all SCIDER.md files from workspace upward.

    Walks from ``workspace_path`` (or cwd) up to the filesystem root, plus
    home dir. At each directory, checks for .scider/SCIDER.md, SCIDER.md,
    and .scider/rules/*.md. All found files are concatenated root-first.

    Returns the assembled string or None if nothing found.
    """
    base = Path(workspace_path).resolve() if workspace_path else Path.cwd().resolve()
    cache_key = str(base)

    if cache_key in _cache:
        return _cache[cache_key]

    dirs = walk_up_dirs(base)

    fragments: list[str] = []
    for d in dirs:
        _collect_from_dir(d, fragments)

    if not fragments:
        _cache[cache_key] = None
        return None

    assembled = f"{_INSTRUCTION_HEADER}\n\n" + "\n\n".join(fragments)

    if len(assembled) > MAX_SCIDER_MD_CHARS:
        assembled = assembled[:MAX_SCIDER_MD_CHARS] + "\n\n[Truncated]"

    _cache[cache_key] = assembled
    return assembled


def _collect_from_dir(directory: Path, fragments: list[str]) -> None:
    """Check a single directory for SCIDER.md candidates and append to fragments."""
    for candidate in _CANDIDATES:
        path = directory / candidate
        content = _read_file(path)
        if content:
            fragments.append(f"Contents of {path}:\n\n{content}")

    rules_dir = directory / ".scider" / "rules"
    if rules_dir.is_dir():
        for path in sorted(rules_dir.glob("*.md")):
            content = _read_file(path)
            if content:
                fragments.append(f"Contents of {path}:\n\n{content}")


def _read_file(path: Path) -> str | None:
    """Read a single file, return stripped content or None."""
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return None
        logger.debug("Loaded SCIDER context from {}", path)
        return content
    except Exception as e:
        logger.warning("Failed to read {}: {}", path, e)
        return None


def clear_cache() -> None:
    """Clear the cached content (useful for testing)."""
    _cache.clear()
