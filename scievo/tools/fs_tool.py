import os
import re
import stat
from datetime import datetime
from pathlib import Path

import filetype

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
FILE_CHUNK_SIZE = 16000  # Maximum bytes to read in a single call


class fstoolUtils:
    TEXT_TYPES = [
        "txt",
        "md",
        "csv",
        "log",
        "json",
        "jsonl",
        "xml",
        "html",
        "htm",
        "yaml",
        "yml",
        "ini",
        "cfg",
        "py",
        "java",
        "c",
        "cpp",
        "h",
        "sh",
        "bash",
        "bat",
        "rtf",
        "toml",
    ]

    @staticmethod
    def add_line_numbers(text: str) -> str:
        """Add line numbers to the given text."""
        lines = text.splitlines()
        width = len(str(len(lines)))
        numbered_lines = [f"{i + 1:>{width}}: {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)

    @staticmethod
    def guess_file_type(path: str) -> tuple[str | None, str | None]:
        """Guess the file type based on its content and extension.

        Returns a tuple of (extension, mime type) or (None, None) if unknown.
        """
        type_guess_ext = None
        type_guess_mime = None
        if (g := filetype.guess(path)) is not None:
            type_guess_ext = g.extension
            type_guess_mime = g.mime
        elif (suffix := Path(path).suffix.lstrip(".").lower()) in fstoolUtils.TEXT_TYPES:
            type_guess_ext = suffix
            match suffix:
                case "md" | "markdown":
                    type_guess_mime = "text/markdown"
                case "csv":
                    type_guess_mime = "text/csv"
                case "json":
                    type_guess_mime = "application/json"
                case "xml":
                    type_guess_mime = "application/xml"
                case "html" | "htm":
                    type_guess_mime = "text/html"
                case _:
                    type_guess_mime = "text/plain"

        return type_guess_ext, type_guess_mime


@register_tool(
    "fs",
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Produce an 'ls -l' style listing (non-recursive) for the given path with pagination support.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Filesystem path to inspect",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of entries to skip from the start (for pagination)",
                        "default": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of entries to return (default at 50 for normal case, and capped at 100)",
                        "default": 50,
                    },
                },
                "required": ["path"],
            },
        },
    },
)
def list_files(path: str, offset: int = 0, limit: int = 50) -> str:
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

    # Validate and normalize pagination parameters
    total_entries = len(records)
    if offset is None or offset < 0:
        offset = 0
    elif offset > total_entries:
        offset = total_entries

    if limit is None or limit <= 0:
        limit = 50

    # Cap limit at 100 entries
    limit = min(limit, 100)

    # Extract paginated records
    paginated_records = records[offset : offset + limit]
    next_offset = offset + len(paginated_records)
    has_more = next_offset < total_entries

    # Calculate column widths only for displayed records
    if paginated_records:
        link_width = max((len(str(r["nlink"])) for r in paginated_records), default=1)
        owner_width = max((len(r["owner"]) for r in paginated_records), default=1)
        group_width = max((len(r["group"]) for r in paginated_records), default=1)
        size_width = max((len(str(r["size"])) for r in paginated_records), default=1)
    else:
        link_width = owner_width = group_width = size_width = 1

    # Format output with pagination metadata
    lines: list[str] = []

    # Add pagination header
    lines.append(f"[Path: {path}]")
    lines.append(f"[Total entries: {total_entries}]")
    lines.append(f"[Current offset: {offset}]")
    lines.append(f"[Limit: {limit}]")
    lines.append(f"[Returned entries: {len(paginated_records)}]")
    lines.append(f"[Has more: {has_more}]")
    if has_more:
        lines.append(f"[Next offset: {next_offset}]")
    lines.append("---")

    # Add total blocks line
    if paginated_records and show_total:
        lines.append(f"total {total_blocks}")

    # Add file entries
    for r in paginated_records:
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
        path = os.path.expandvars(os.path.expanduser(path))
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
            "description": "Read file content with pagination support (UTF-8 encoding). Returns content and metadata. Each call is limited to returning at most `max_char` bytes. Use offset to read subsequent pages.",
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
                        "description": f"Maximum number of characters to read (default at 8000 for normal case, capped at {FILE_CHUNK_SIZE})",
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
    - File content (up to max_char bytes from offset, capped at FILE_CHUNK_SIZE)
    - Total file size
    - Current offset position
    - Whether more content is available

    Non-UTF-8 characters are replaced using the 'replace' error handler.
    For files larger than max_char, use the offset parameter to read subsequent pages.
    """
    path = os.path.expandvars(os.path.expanduser(path))
    # Cap max_char at 32000 bytes
    max_char = min(max_char, FILE_CHUNK_SIZE)

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

        file_ext, file_mime = fstoolUtils.guess_file_type(path)
        if file_ext and file_mime:
            result_parts.append(f"[Guessed type: {file_ext}, mime: {file_mime}]")
            if filetype.is_image(path):
                texts = [
                    "[Note: Binary image content not displayed. Use appropriate tools to view images.]"
                ]
                try:
                    from PIL import Image

                    with Image.open(path) as img:
                        width, height = img.size
                    texts += f"\n[Image dimensions: {width}x{height}, format: {img.format}]"
                except:
                    pass

                out_content = "\n".join(texts)
            elif filetype.is_video(path):
                out_content = "[Note: Binary video content not displayed. Use appropriate tools to play videos.]"
            elif filetype.is_audio(path):
                out_content = "[Note: Binary audio content not displayed. Use appropriate tools to play audio files.]"
            elif filetype.is_archive(path):
                out_content = "[Note: Binary archive content not displayed. Use appropriate tools to extract archives.]"
            elif file_ext in fstoolUtils.TEXT_TYPES:
                decoded_content = content.decode("utf-8", errors="replace")
                out_content = fstoolUtils.add_line_numbers(decoded_content)
        else:
            out_content = content.decode("utf-8", errors="replace")

        if has_more:
            result_parts.append(f"[Next offset: {next_offset} bytes]")
            out_content += " (...truncated...)"

        result_parts.append("---")
        result_parts.append(out_content)

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
    path = os.path.expandvars(os.path.expanduser(path))
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
        path = os.path.expandvars(os.path.expanduser(path))
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return f"Directory ensured: '{str(p.resolve())}'"
    except Exception as e:
        return f"Error creating directory '{path}': {e}"


EDIT_FILE_DESC = """\
# Overview of the edit file tool.
- use the edit_file tool when you want to make quick changes in the form of a unified diff.

## Details on using the edit_file tool:
    Each call to the edit_file tool must include the following keys:
    - file_path (str, required): the path to the file to edit
    - unified_diff (str, required): a single unified diff string to apply to the file.
        - Unified diffs start with a hunk header: @@ -<start_old>,<len_old> +<start_new>,<len_new> @@
        - Lines starting with '-' indicate deletions
        - Lines starting with '+' indicate additions
        - Lines starting with ' ' (space) indicate context (unchanged lines)

## Examples of unified diff tool calls. The unified diff can be as long as you want. Including content (unchanged lines) can be useful to help the tool apply the diff correctly.
<example_unified_diff_tool_calls>
1. Adding new lines to a file (Python):
    {
    "file_path": "src/utils.py",
    "unified_diff": "@@ -10,6 +10,9 @@\n def existing_function():\n     # Some existing code\n     return result\n+\n+def new_function():\n+    return 'This is a new function'\n \n # More existing code"
    }

2. Modifying existing lines (TypeScript):
    {
    "file_path": "src/services/userService.ts",
    "unified_diff": "@@ -15,7 +15,7 @@\n class UserService <bracket>\n   private logger: Logger;\n   private timeout: number;\n-  constructor(private apiClient: ApiClient, timeout: number = 30) <bracket>\n+  constructor(private apiClient: ApiClient, timeout: number = 60) <bracket>\n     this.logger = new Logger('UserService');\n     this.timeout = timeout;\n   </bracket>"
    }

3. Deleting lines (React/JSX):
    {
    "file_path": "src/components/DataDisplay.jsx",
    "unified_diff": "@@ -22,9 +22,6 @@\n   const processData = (data) => <bracket>\n     // Process the data\n     const result = transform(data);\n-\n-    // This debug code is no longer needed\n-    console.log('Debug:', result);\n \n     return result;\n   </bracket>;"
    }

4. Using a large unified diff for multiple changes (JSON):
    {
    "file_path": "config/settings.json",
    "unified_diff": "@@ -5,6 +5,11 @@\n   \"environment\": \"development\",\n   \"logLevel\": \"debug\",\n   \"database\": <bracket>\n+    \"host\": \"localhost\",\n+    \"port\": 5432,\n+    \"username\": \"admin\",\n+    \"password\": \"secure_password\",\n+    \"name\": \"app_db\"\n   </bracket>,\n@@ -25,6 +30,10 @@\n   \"api\": <bracket>\n     \"baseUrl\": \"http://localhost:8000\",\n     \"timeout\": 30000\n+  </bracket>,\n+  \"cache\": <bracket>\n+    \"enabled\": true,\n+    \"ttl\": 3600\n   </bracket>\n </bracket>"
    }
</example_unified_diff_tool_calls>
"""


@register_tool(
    "fs",
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": EDIT_FILE_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "unified_diff": {
                        "type": "string",
                        "description": "Unified diff string to apply to the file. See tool description for details.",
                    },
                },
                "required": ["file_path", "unified_diff"],
            },
        },
    },
)
def edit_file(file_path: str, unified_diff: str) -> str:
    """
    Applies a unified diff patch (for a single file) to file_path.

    Returns True if the patch was applied successfully, False otherwise.
    """
    # Read the original file lines; if the file doesn't exist, treat it as empty.
    path = Path(file_path)
    if path.exists():
        original_lines = path.read_text(encoding="utf8").splitlines(keepends=True)
    else:
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    new_lines = []
    current_index = 0

    ERR_PREFIX = "Error applying unified diff patch: "

    patch_lines = unified_diff.splitlines()
    # Regex for a hunk header, e.g., @@ -3,7 +3,6 @@
    hunk_header_re = re.compile(r"^@@(?: -(\d+)(?:,(\d+))?)?(?: \+(\d+)(?:,(\d+))?)? @@")
    i = 0
    while i < len(patch_lines):
        line = patch_lines[i]
        if line.lstrip().startswith("@@"):
            if line.strip() == "@@":
                # Handle minimal hunk header without line numbers.
                orig_start = 1
            else:
                m = hunk_header_re.match(line.strip())
                if not m:
                    raise ValueError(ERR_PREFIX + "Invalid hunk header: " + line)
                orig_start = int(m.group(1)) if m.group(1) is not None else 1
            hunk_start_index = orig_start - 1  # diff headers are 1-indexed
            if hunk_start_index > len(original_lines):
                raise ValueError(ERR_PREFIX + "Hunk start index beyond file length")
            new_lines.extend(original_lines[current_index:hunk_start_index])
            current_index = hunk_start_index
            i += 1
            # Process the hunk lines until the next hunk header.
            while i < len(patch_lines) and not patch_lines[i].startswith("@@"):
                pline = patch_lines[i]
                if pline.startswith(" "):
                    # Context line must match exactly.
                    expected = pline[1:]
                    if current_index >= len(original_lines):
                        raise ValueError(ERR_PREFIX + "Context line expected but file ended")
                    orig_line = original_lines[current_index].rstrip("\n")
                    if orig_line != expected:
                        raise ValueError(
                            ERR_PREFIX
                            + f"Context line mismatch. Expected: {expected}. Got: {orig_line}"
                        )
                    new_lines.append(original_lines[current_index])
                    current_index += 1
                elif pline.startswith("-"):
                    # Removal line: verify and skip from original.
                    expected = pline[1:]
                    if current_index >= len(original_lines):
                        raise ValueError(ERR_PREFIX + "Removal line expected but file ended")
                    orig_line = original_lines[current_index].rstrip("\n")
                    if orig_line != expected:
                        raise ValueError(
                            ERR_PREFIX
                            + f"Removal line mismatch. Expected: {expected}. Got: {orig_line}"
                        )
                    current_index += 1
                elif pline.startswith("+"):
                    # Addition line: add to new_lines.
                    new_lines.append(pline[1:] + "\n")
                else:
                    try:
                        expected = pline
                        if current_index >= len(original_lines):
                            raise ValueError(
                                ERR_PREFIX + "We are trying a smart diff, dumb diff failed"
                            )
                        orig_line = original_lines[current_index].rstrip("\n")
                        if orig_line != expected:
                            raise ValueError(
                                ERR_PREFIX + "We are trying a smart diff, dumb diff failed"
                            )
                        new_lines.append(original_lines[current_index])
                        current_index += 1
                    except Exception as e:
                        raise ValueError(
                            ERR_PREFIX + "We are trying a smart diff, dumb diff failed"
                        ) from e
                i += 1
        else:
            # Skip non-hunk header lines.
            i += 1

    # Append any remaining lines from the original file.
    new_lines.extend(original_lines[current_index:])
    # Write the new content back to the file. Ensure the file ends with a newline
    # to match typical patch behavior and avoid tooling conflicts.
    content = "".join(new_lines)
    if content and not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf8")
    return "The patch was applied successfully."


@register_tool(
    "fs",
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a single file from the filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to delete",
                    },
                },
                "required": ["path"],
            },
        },
    },
)
def delete_file(path: str) -> str:
    """Delete a single file from the filesystem."""
    path = os.path.expandvars(os.path.expanduser(path))
    try:
        if not os.path.exists(path):
            return f"Error: File '{path}' does not exist."
        if os.path.isdir(path):
            return f"Error: '{path}' is a directory, not a file. Use appropriate tools to remove directories."
        os.remove(path)
        return f"Successfully deleted file: '{path}'"
    except PermissionError:
        return f"Error: Permission denied to delete '{path}'."
    except Exception as e:
        return f"Error deleting file '{path}': {e}"


if __name__ == "__main__":
    # test
    text = """\
This is line 1.
This is line 2.
This is line 3.
"""
    diff = """\
@@ -1,3 +1,3 @@
-This is line 1.
+This is a new line 1.5.
 This is line 2.
 This is line 3.
"""
    import tempfile

    with tempfile.NamedTemporaryFile("w+", delete=True) as tf:
        tf.write(text)
        tf.flush()
        tf_path = tf.name
        result = edit_file(tf_path, diff)
        if result:
            with open(tf_path, "r") as f:
                print("Patched file content:")
                print(f.read())
