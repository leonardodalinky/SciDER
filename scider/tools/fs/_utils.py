"""Shared utilities for filesystem tools."""

from pathlib import Path

import filetype

# Maximum bytes to read in a single call
FILE_CHUNK_SIZE = 16000

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


def add_line_numbers(text: str) -> str:
    """Add line numbers to the given text."""
    lines = text.splitlines()
    width = len(str(len(lines)))
    numbered_lines = [f"{i + 1:>{width}}: {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


def guess_file_type(path: str) -> tuple[str | None, str | None]:
    """Guess the file type based on its content and extension.

    Returns a tuple of (extension, mime type) or (None, None) if unknown.
    """
    type_guess_ext = None
    type_guess_mime = None
    if (g := filetype.guess(path)) is not None:
        type_guess_ext = g.extension
        type_guess_mime = g.mime
    elif (suffix := Path(path).suffix.lstrip(".").lower()) in TEXT_TYPES:
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
