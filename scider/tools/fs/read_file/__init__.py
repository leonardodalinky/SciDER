"""ReadFileTool — read file content with line-based pagination.

Modeled after Claude Code's FileReadTool. Key features:
- Line-based offset/limit (not byte-based) for precise navigation
- Line numbers prepended to output (N\\tLINE format)
- Binary file detection (images, video, audio, archives)
- Default limit of 2000 lines, configurable per call
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext
from .._utils import TEXT_TYPES, guess_file_type

# Default and maximum line limits
DEFAULT_LINE_LIMIT = 2000
MAX_LINE_LIMIT = 10000
# Max file size allowed without explicit offset+limit (256 KB, same as Claude Code)
MAX_FILE_SIZE_BYTES = 256 * 1024
# Max file size for streaming read path (10 MB)
MAX_STREAM_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Binary extensions that should not be read as text
BINARY_EXTENSIONS = {
    "exe",
    "dll",
    "so",
    "dylib",
    "bin",
    "obj",
    "o",
    "a",
    "lib",
    "zip",
    "tar",
    "gz",
    "bz2",
    "xz",
    "7z",
    "rar",
    "mp3",
    "mp4",
    "avi",
    "mkv",
    "mov",
    "flac",
    "wav",
    "ogg",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "webp",
    "bmp",
    "ico",
    "tiff",
    "svg",
    "pdf",
    "doc",
    "docx",
    "xls",
    "xlsx",
    "ppt",
    "pptx",
    "pyc",
    "pyo",
    "class",
    "wasm",
}

# Image extensions that get special handling
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"}


class ReadFileInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to read")
    offset: int | None = Field(
        default=None,
        description="Line number to start reading from (1-indexed). Defaults to 1.",
    )
    limit: int | None = Field(
        default=None,
        description=(
            f"Number of lines to read. Defaults to {DEFAULT_LINE_LIMIT}. " f"Max {MAX_LINE_LIMIT}."
        ),
    )


class ReadFileTool(BaseTool):
    name = "Read"
    description = (
        "Read a file from the filesystem. "
        "Output includes line numbers prefixed as 'N\\tLINE'. "
        "Use offset and limit for large files. "
        "Binary files (images, archives, etc.) return metadata only."
    )
    input_schema = ReadFileInput
    _always_read_only = True
    # Never persist read_file output — would cause circular reads
    max_result_size_chars = float("inf")
    prompt = (
        "- Read files with the read_file tool. Output has line numbers.\n"
        "- For large files, use offset and limit to read specific sections.\n"
        "- offset is 1-indexed (first line is 1). Default limit is 2000 lines.\n"
        "- Binary files (images, archives) return metadata, not content."
    )

    def call(
        self,
        context: ToolContext,
        *,
        file_path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        file_path = os.path.expandvars(os.path.expanduser(file_path))

        # Validate file exists
        if not os.path.exists(file_path):
            suggestion = _suggest_similar_file(file_path)
            msg = f"Error: File '{file_path}' does not exist."
            if suggestion:
                msg += f" Did you mean '{suggestion}'?"
            return msg

        if os.path.isdir(file_path):
            return f"Error: '{file_path}' is a directory, not a file. Use list_files instead."

        # Check for binary files
        ext = Path(file_path).suffix.lstrip(".").lower()
        if ext in BINARY_EXTENSIONS:
            return _handle_binary_file(file_path, ext)

        # Normalize offset/limit
        start_line = max((offset or 1) - 1, 0)  # Convert 1-indexed to 0-indexed
        max_lines = min(limit or DEFAULT_LINE_LIMIT, MAX_LINE_LIMIT)

        try:
            file_size = os.path.getsize(file_path)

            # Guard: large files without explicit range should fail early.
            # This yields a ~100-byte error (cheap) instead of ~25K tokens of
            # content, matching Claude Code's strategy from limits.ts.
            if file_size > MAX_FILE_SIZE_BYTES and offset is None and limit is None:
                return (
                    f"Error: File '{file_path}' is too large "
                    f"({file_size:,} bytes, max {MAX_FILE_SIZE_BYTES:,} bytes). "
                    f"Use offset and limit to read specific portions. "
                    f"For example: read_file(file_path='{file_path}', offset=1, limit=200)"
                )

            if file_size > MAX_STREAM_FILE_SIZE_BYTES:
                # Streaming read for large files
                return _read_large_file(file_path, start_line, max_lines, file_size)

            # Fast path: read entire file into memory
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)
            selected = all_lines[start_line : start_line + max_lines]
            num_lines = len(selected)

            if num_lines == 0:
                return (
                    f"[File: {file_path}]\n"
                    f"[Total lines: {total_lines}]\n"
                    f"[No content at offset {start_line + 1}]"
                )

            # Add line numbers: "N\tLINE"
            numbered = _add_line_numbers(selected, start_line + 1)

            # Build result
            parts = [f"[File: {file_path}]"]
            parts.append(f"[Lines: {start_line + 1}-{start_line + num_lines} of {total_lines}]")

            if start_line + num_lines < total_lines:
                remaining = total_lines - (start_line + num_lines)
                parts.append(
                    f"[{remaining} more lines. Use offset={start_line + num_lines + 1} to continue]"
                )

            parts.append("")
            parts.append(numbered)

            return "\n".join(parts)

        except Exception as e:
            return f"Error reading file '{file_path}': {e}"


def _add_line_numbers(lines: list[str], start: int) -> str:
    """Add line numbers in compact 'N\\tLINE' format."""
    result = []
    for i, line in enumerate(lines):
        line_content = line.rstrip("\n").rstrip("\r")
        result.append(f"{start + i}\t{line_content}")
    return "\n".join(result)


def _read_large_file(file_path: str, start_line: int, max_lines: int, file_size: int) -> str:
    """Stream-read a large file, only keeping lines in range."""
    lines_read = []
    total_lines = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f):
            total_lines = line_num + 1
            if line_num >= start_line and len(lines_read) < max_lines:
                lines_read.append(line)
            # Continue counting total lines even after we have enough
            if len(lines_read) >= max_lines and line_num > start_line + max_lines + 1000:
                # Estimate remaining lines for very large files
                bytes_read = f.tell()
                if bytes_read > 0:
                    avg_line_len = bytes_read / total_lines
                    total_lines = int(file_size / avg_line_len)
                break

    num_lines = len(lines_read)
    if num_lines == 0:
        return (
            f"[File: {file_path}]\n"
            f"[Total lines: ~{total_lines}]\n"
            f"[No content at offset {start_line + 1}]"
        )

    numbered = _add_line_numbers(lines_read, start_line + 1)

    parts = [f"[File: {file_path}]"]
    parts.append(f"[Lines: {start_line + 1}-{start_line + num_lines} of ~{total_lines}]")
    parts.append(f"[File size: {file_size:,} bytes]")

    if start_line + num_lines < total_lines:
        remaining = total_lines - (start_line + num_lines)
        parts.append(
            f"[~{remaining} more lines. Use offset={start_line + num_lines + 1} to continue]"
        )

    parts.append("")
    parts.append(numbered)

    return "\n".join(parts)


def _handle_binary_file(file_path: str, ext: str) -> str:
    """Return metadata for binary files."""
    file_size = os.path.getsize(file_path)
    parts = [
        f"[File: {file_path}]",
        f"[Size: {file_size:,} bytes]",
        f"[Type: {ext}]",
    ]

    if ext in IMAGE_EXTENSIONS:
        parts.append("[Binary image file]")
        try:
            from PIL import Image

            with Image.open(file_path) as img:
                parts.append(f"[Dimensions: {img.size[0]}x{img.size[1]}, format: {img.format}]")
        except Exception:
            pass
    elif ext == "pdf":
        parts.append("[PDF document — use appropriate PDF tools to read]")
    elif ext in {"zip", "tar", "gz", "bz2", "xz", "7z", "rar"}:
        parts.append("[Archive file — use appropriate tools to extract]")
    elif ext in {"mp3", "mp4", "avi", "mkv", "mov", "flac", "wav", "ogg"}:
        parts.append("[Media file — binary content not displayed]")
    else:
        parts.append("[Binary file — content not displayed]")

    return "\n".join(parts)


def _suggest_similar_file(file_path: str) -> str | None:
    """Suggest a similar file if the given path doesn't exist."""
    parent = os.path.dirname(file_path)
    name = os.path.basename(file_path)

    if not parent or not os.path.isdir(parent):
        return None

    try:
        candidates = os.listdir(parent)
    except OSError:
        return None

    # Case-insensitive match
    name_lower = name.lower()
    for c in candidates:
        if c.lower() == name_lower:
            return os.path.join(parent, c)

    return None
