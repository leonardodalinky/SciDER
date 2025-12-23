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

from .registry import register_tool, register_toolset_desc

register_toolset_desc("fs", "File system toolset.")

# Configuration
FILE_CHUNK_SIZE = 32000  # Maximum bytes to read in a single call


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
                    "path": {
                        "type": "string",
                        "description": "Filesystem path to inspect",
                    }
                },
                "required": ["path"],
            },
        },
    },
)
def list_files(path: str) -> str:
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
            "description": "Read the first N lines of multiple files (default 10 lines each). Truncate output to 2000 characters per file if longer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to read (at least one required)",
                        "minItems": 1,
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of lines to read from the head of each file",
                        "default": 10,
                    },
                },
                "required": ["paths"],
            },
        },
    },
)
def read_head(paths: list[str], n: int = 10) -> str:
    if not paths:
        return "Error: At least one path must be provided"

    def _read_single_file(path: str, n: int) -> str:
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

    results = []
    for path in paths:
        file_content = _read_single_file(path, n)
        results.append(f"=== {path} ===")
        results.append(file_content)
        results.append(f"=== End of {path} ===")

    return "\n".join(results).rstrip()


@register_tool(
    "fs",
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file content with pagination support (UTF-8 encoding). Returns content and metadata. Each call is limited to returning at most max_char bytes (capped at 32000). Use offset to read subsequent pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "offset": {
                        "type": "integer",
                        "description": "Number of bytes to skip from the start (for pagination)",
                        "default": 0,
                    },
                    "max_char": {
                        "type": "integer",
                        "description": "Maximum number of characters to read (will be capped at 32000)",
                        "default": 8000,
                    },
                },
                "required": ["path"],
            },
        },
    },
)
def read_file(path: str, offset: int = 0, max_char: int = 8000) -> str:
    """
    Read file content with pagination support (UTF-8 encoding).

    Returns a formatted string containing:
    - File content (up to max_char bytes from offset, capped at 32000)
    - Total file size
    - Current offset position
    - Whether more content is available

    Non-UTF-8 characters are replaced using the 'replace' error handler.
    For files larger than max_char, use the offset parameter to read subsequent pages.
    """
    # Cap max_char at 32000 bytes
    max_char = min(max_char, 32000)

    try:
        # Get file size without reading entire file
        file_size = os.path.getsize(path)

        # Validate and normalize offset
        if offset is None or offset < 0:
            offset = 0
        elif offset > file_size:
            offset = file_size

        # Read only the needed chunk from disk with UTF-8 encoding
        with open(path, "rb") as f:
            f.seek(offset)
            content = f.read(max_char)

        next_offset = offset + max_char
        has_more = next_offset < file_size

        # Format pagination metadata
        result_parts = [
            f"[File: {path}]",
            f"[Total size: {file_size} bytes]",
            f"[Current offset: {offset} bytes]",
            f"[Max chunk size: {max_char} bytes]",
            f"[Actual chunk size: {len(content)} bytes]",
            f"[Has more content: {has_more}]",
        ]

        if has_more:
            result_parts.append(f"[Next offset: {next_offset} bytes]")

        result_parts.append("---")
        result_parts.append(content.decode("utf-8", errors="replace"))

        return "\n".join(result_parts)
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
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
)
def save_file(path: str, content: str) -> str:
    dir_path = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(dir_path):
        return f"Error: Directory '{dir_path}' does not exist. Please create it first using the 'create_dir' tool."
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
                    "path": {
                        "type": "string",
                        "description": "Directory path to create",
                    },
                },
                "required": ["path"],
            },
        },
    },
)
def create_dir(path: str) -> str:
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return f"Directory ensured: '{str(p.resolve())}'"
    except Exception as e:
        return f"Error creating directory '{path}': {e}"
