"""ReadFileTool — read file content with line-based pagination.

Modeled after Claude Code's FileReadTool. Key features:
- Line-based offset/limit (not byte-based) for precise navigation
- Line numbers prepended to output (N\\tLINE format)
- Binary file detection (images, video, audio, archives)
- Default limit of 2000 lines, configurable per call
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext, ToolImage, ToolResult
from .._utils import TEXT_TYPES, guess_file_type

# Default and maximum line limits
DEFAULT_LINE_LIMIT = 2000
MAX_LINE_LIMIT = 10000
# Max file size allowed without explicit offset+limit (256 KB, same as Claude Code)
MAX_FILE_SIZE_BYTES = 256 * 1024
# Max file size for streaming read path (10 MB)
MAX_STREAM_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Image payload limit — payload larger than this is compressed before being
# attached to the tool result. Matches typical per-image limits of Claude and
# Gemini APIs.
MAX_IMAGE_PAYLOAD_BYTES = 5 * 1024 * 1024

# Extensions SciDER knows how to encode and send as an image to the LLM.
# Other image extensions (BMP, TIFF, SVG, ICO) fall back to text metadata.
SUPPORTED_IMAGE_MIME: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}

# JPEG quality ladder for compression — tried in order until payload fits.
_JPEG_QUALITY_LADDER = [90, 80, 70, 60, 50, 40]

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
        "Text files: output includes line numbers prefixed as 'N\\tLINE'; "
        "use offset and limit for large files. "
        "Image files (PNG/JPEG/WebP/GIF): if your model supports vision, "
        "the actual image is attached to the tool result so you can see it; "
        "oversized images are auto-compressed. "
        "Other binary files (archives, PDFs, audio, video) return text metadata only."
    )
    input_schema = ReadFileInput
    _always_read_only = True
    # Never persist read_file output — would cause circular reads
    max_result_size_chars = float("inf")
    prompt = (
        "# Read tool usage\n"
        "- Use Read to read files instead of Bash with cat/head/tail.\n"
        "- Output includes line numbers. For large files, use `offset` and `limit` to read specific sections.\n"
        "- `offset` is 1-indexed (first line is 1). Default limit is 2000 lines.\n"
        "- Files larger than 256KB are rejected — use offset/limit to read in sections.\n"
        "- **Images** (PNG, JPEG, WebP, GIF): if your model has vision, Read attaches "
        "the actual image to the tool result and you will see its visual content alongside "
        "text metadata (dimensions, file size). Oversized images (>5 MB) are auto-compressed. "
        "Use this to inspect plots, figures, screenshots, and any image data in the workspace.\n"
        "- Other binary files (archives, PDFs, media) return text metadata only — no content.\n"
    )

    def call(
        self,
        context: ToolContext,
        *,
        file_path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> "str | ToolResult":
        file_path = os.path.expandvars(os.path.expanduser(file_path))

        # Validate file exists
        if not os.path.exists(file_path):
            suggestion = _suggest_similar_file(file_path)
            msg = f"Error: File '{file_path}' does not exist."
            if suggestion:
                msg += f" Did you mean '{suggestion}'?"
            return msg

        if os.path.isdir(file_path):
            # List first few entries to help the model pick a file
            try:
                entries = sorted(os.listdir(file_path))[:20]
                listing = "\n".join(
                    f"  {e}/" if os.path.isdir(os.path.join(file_path, e)) else f"  {e}"
                    for e in entries
                )
                if len(os.listdir(file_path)) > 20:
                    listing += f"\n  ... and {len(os.listdir(file_path)) - 20} more"
                return (
                    f"Error: '{file_path}' is a directory, not a file. "
                    f"Use Glob to find files by pattern, or read a specific file.\n"
                    f"Directory contents:\n{listing}"
                )
            except Exception:
                return f"Error: '{file_path}' is a directory, not a file. Use Glob to find files by pattern."

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
                    f"For data files, read a small SAMPLE (e.g. `limit=50`) to inspect the schema, "
                    f"then use `Bash` with `wc -l`, `awk`, or `python3 -c '...'` to compute statistics "
                    f"on the full file. Do NOT paginate through the entire file with Read."
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


def _handle_binary_file(file_path: str, ext: str) -> "str | ToolResult":
    """Return metadata for binary files.

    For supported image formats (PNG/JPEG/WebP/GIF) the result is a
    ``ToolResult`` carrying both text metadata and a base64-encoded image
    block so the LLM can actually see the image. All other binary formats
    return a text-only metadata string as before.
    """
    if ext in SUPPORTED_IMAGE_MIME:
        return _handle_image_file(file_path, ext)

    file_size = os.path.getsize(file_path)
    parts = [
        f"[File: {file_path}]",
        f"[Size: {file_size:,} bytes]",
        f"[Type: {ext}]",
    ]

    if ext in IMAGE_EXTENSIONS:
        # Known image format we don't pass to the LLM (BMP, TIFF, SVG, ICO, ...)
        parts.append("[Binary image file]")
        parts.append("[Format not supported for model viewing — metadata only]")
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


def _handle_image_file(file_path: str, ext: str) -> "ToolResult | str":
    """Read an image file, compress if needed, and return a ToolResult.

    The tool result includes both a text metadata block (so the model can
    cite dimensions and file size) and a base64-encoded image block.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed — falling back to text metadata for image")
        return _image_metadata_text(file_path, ext, extra="[Pillow not installed]")

    file_size = os.path.getsize(file_path)
    try:
        raw_bytes = Path(file_path).read_bytes()
    except Exception as e:
        return _image_metadata_text(file_path, ext, extra=f"[Failed to read: {e}]")

    # Collect dimensions via PIL for the metadata block; do not fail if it doesn't open.
    dims_text = ""
    pil_format: str | None = None
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            dims_text = f"[Dimensions: {img.size[0]}x{img.size[1]}, format: {img.format}]"
            pil_format = img.format
    except Exception as e:
        logger.debug("PIL could not read {}: {}", file_path, e)

    media_type = SUPPORTED_IMAGE_MIME[ext]
    payload_bytes = raw_bytes
    compression_note = ""

    if len(payload_bytes) > MAX_IMAGE_PAYLOAD_BYTES:
        compressed = _compress_image_to_limit(raw_bytes, MAX_IMAGE_PAYLOAD_BYTES)
        if compressed is None:
            return _image_metadata_text(
                file_path,
                ext,
                extra=(
                    f"[Image {file_size:,} bytes — could not compress under "
                    f"{MAX_IMAGE_PAYLOAD_BYTES:,} bytes, not attached]"
                ),
            )
        payload_bytes, media_type = compressed
        compression_note = (
            f"[Compressed from {file_size:,} to {len(payload_bytes):,} bytes "
            f"({media_type}) to fit under the {MAX_IMAGE_PAYLOAD_BYTES:,}-byte limit]"
        )

    b64 = base64.b64encode(payload_bytes).decode("ascii")

    text_parts = [
        f"[File: {file_path}]",
        f"[Size: {file_size:,} bytes]",
        f"[Type: {ext}]",
    ]
    if dims_text:
        text_parts.append(dims_text)
    if compression_note:
        text_parts.append(compression_note)
    text_parts.append("[Image attached for model viewing]")

    return ToolResult(
        text="\n".join(text_parts),
        images=[ToolImage(media_type=media_type, data=b64)],
    )


def _image_metadata_text(file_path: str, ext: str, *, extra: str = "") -> str:
    """Text-only fallback when an image cannot be attached."""
    try:
        file_size = os.path.getsize(file_path)
    except OSError:
        file_size = 0
    parts = [
        f"[File: {file_path}]",
        f"[Size: {file_size:,} bytes]",
        f"[Type: {ext}]",
    ]
    if extra:
        parts.append(extra)
    return "\n".join(parts)


def _compress_image_to_limit(
    raw_bytes: bytes,
    limit: int,
) -> tuple[bytes, str] | None:
    """Compress an image until its byte size is <= ``limit``.

    Strategy:
    1. Re-encode as JPEG with a decreasing quality ladder.
    2. If JPEG at the lowest quality is still too large, iteratively resize
       the image by 0.8x and retry the quality ladder.
    3. Give up after a handful of resize rounds to avoid pathological loops.

    Returns ``(compressed_bytes, media_type)`` or ``None`` on failure.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.load()
            # Normalize to RGB for JPEG (drops alpha). Acceptable because this
            # path only fires when the raw payload is already too big to send.
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            current = img
            for _ in range(8):  # up to 8 resize rounds → (0.8)^8 ≈ 0.17x
                for quality in _JPEG_QUALITY_LADDER:
                    buf = io.BytesIO()
                    current.save(buf, format="JPEG", quality=quality, optimize=True)
                    data = buf.getvalue()
                    if len(data) <= limit:
                        return data, "image/jpeg"
                # Still too big — shrink and retry.
                new_size = (max(1, int(current.size[0] * 0.8)), max(1, int(current.size[1] * 0.8)))
                if new_size == current.size:
                    break
                current = current.resize(new_size, Image.LANCZOS)
    except Exception as e:
        logger.warning("Image compression failed: {}", e)
        return None

    return None


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
