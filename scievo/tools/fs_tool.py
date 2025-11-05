import os
import stat
from datetime import datetime
from pathlib import Path

try:  # pragma: no cover - platform dependent
    import grp
except ImportError:  # pragma: no cover - Windows fallback
    grp = None  # type: ignore[assignment]

try:  # pragma: no cover - platform dependent
    import pwd
except ImportError:  # pragma: no cover - Windows fallback
    pwd = None  # type: ignore[assignment]

from scievo.core.types import GraphState

from .registry import register_tool, register_toolset_desc

register_toolset_desc("fs", "File system toolset.")


@register_tool(
    "fs",
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Produce an 'ls -l' style listing (non-recursive) for the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Filesystem path to inspect"}
                },
                "required": ["path"],
            },
        },
    },
)
def list_files(graph_state: GraphState, path: str) -> str:
    def _resolve_owner(uid: int) -> str:
        if pwd is None:
            return str(uid)
        try:
            return pwd.getpwuid(uid).pw_name  # type: ignore[attr-defined]
        except KeyError:
            return str(uid)

    def _resolve_group(gid: int) -> str:
        if grp is None:
            return str(gid)
        try:
            return grp.getgrgid(gid).gr_name  # type: ignore[attr-defined]
        except KeyError:
            return str(gid)

    def _format_mtime(ts: float) -> str:
        dt = datetime.fromtimestamp(ts)
        now = datetime.now()
        day = f"{dt.day:2d}"
        if abs((now - dt).days) >= 180:
            return f"{dt:%b} {day}  {dt:%Y}"
        return f"{dt:%b} {day} {dt:%H:%M}"

    normalized = Path(os.path.expandvars(path)).expanduser()
    is_symlink = normalized.is_symlink()

    try:
        if normalized.is_dir() and not is_symlink:
            entries = sorted(normalized.iterdir(), key=lambda p: p.name)
            show_total = True
        else:
            entries = [normalized]
            show_total = False
    except OSError as e:
        return f"Error listing files in '{path}': {e}"

    records = []
    errors = []
    total_blocks = 0

    for entry in entries:
        try:
            stat_result = entry.lstat()
        except FileNotFoundError:
            errors.append(f"Path '{entry}' does not exist.")
            continue
        except OSError as e:
            errors.append(f"Error accessing '{entry}': {e}")
            continue

        mode = stat.filemode(stat_result.st_mode)
        nlink = stat_result.st_nlink
        owner = _resolve_owner(stat_result.st_uid)
        group = _resolve_group(stat_result.st_gid)
        size = stat_result.st_size
        mtime = _format_mtime(stat_result.st_mtime)

        name = entry.name
        if entry.is_symlink():
            try:
                target = os.readlink(entry)
                name = f"{name} -> {target}"
            except OSError:
                name = f"{name} -> <unresolved>"

        blocks = getattr(stat_result, "st_blocks", None)
        if blocks is None:
            blocks = (size + 511) // 512
        total_blocks += blocks

        records.append(
            {
                "mode": mode,
                "nlink": nlink,
                "owner": owner,
                "group": group,
                "size": size,
                "mtime": mtime,
                "name": name,
            }
        )

    if errors and not records:
        return "\n".join(errors)

    if not records and show_total:
        return "total 0"

    link_width = max((len(str(r["nlink"])) for r in records), default=1)
    owner_width = max((len(r["owner"]) for r in records), default=1)
    group_width = max((len(r["group"]) for r in records), default=1)
    size_width = max((len(str(r["size"])) for r in records), default=1)

    lines: list[str] = []
    if records and show_total:
        lines.append(f"total {total_blocks}")

    for r in records:
        lines.append(
            f"{r['mode']} "
            f"{r['nlink']:>{link_width}} "
            f"{r['owner']:<{owner_width}} "
            f"{r['group']:<{group_width}} "
            f"{r['size']:>{size_width}} "
            f"{r['mtime']} "
            f"{r['name']}"
        )

    lines.extend(errors)

    return "\n".join(lines)


@register_tool(
    "fs",
    {
        "type": "function",
        "function": {
            "name": "read_head",
            "description": "Read the first N lines of a file (default 10). Truncate output to 2000 characters if longer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "n": {
                        "type": "integer",
                        "description": "Number of lines to read from the head",
                        "default": 10,
                    },
                },
                "required": ["path"],
            },
        },
    },
)
def read_head(graph_state: GraphState, path: str, n: int = 10) -> str:
    try:
        lines = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(max(0, n)):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
        text = "".join(lines)
        if len(text) > 2000:
            text = text[:2000]
        return text
    except Exception as e:
        return f"Error reading file '{path}': {e}"


@register_tool(
    "fs",
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the entire file content. Truncate to max_char (default 32000) if longer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "max_char": {
                        "type": "integer",
                        "description": "Maximum number of characters to return",
                        "default": 32000,
                    },
                },
                "required": ["path"],
            },
        },
    },
)
def read_file(graph_state: GraphState, path: str, max_char: int = 32000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if max_char is not None and max_char >= 0 and len(text) > max_char:
            text = text[:max_char]
        return text
    except Exception as e:
        return f"Error reading file '{path}': {e}"


@register_tool(
    "fs",
    {
        "type": "function",
        "function": {
            "name": "save_file",
            "description": "Save the given content to a file path (overwrites existing file).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                },
                "required": ["path", "content"],
            },
        },
    },
)
def save_file(graph_state: GraphState, path: str, content: str) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Saved {len(content)} characters to '{path}'"
    except Exception as e:
        return f"Error saving file '{path}': {e}"


@register_tool(
    "fs",
    {
        "type": "function",
        "function": {
            "name": "create_dir",
            "description": "Create a directory (including parents). No error if it already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to create"},
                },
                "required": ["path"],
            },
        },
    },
)
def create_dir(graph_state: dict, path: str) -> str:
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return f"Directory ensured: '{str(p.resolve())}'"
    except Exception as e:
        return f"Error creating directory '{path}': {e}"
