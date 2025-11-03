import os
from pathlib import Path

from ..core.types import GraphState
from ..core.utils import wrap_dict_to_toon
from . import register_tool


@register_tool(
    "fs",
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files (non-recursive) in the given directory path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list files from"}
                },
                "required": ["path"],
            },
        },
    },
)
def list_files(graph_state: GraphState, path: str) -> str:
    try:

        def human_size(nbytes: int) -> str:
            units = ["B", "KB", "MB", "GB", "TB", "PB"]
            i = 0
            f = float(nbytes)
            while f >= 1024.0 and i < len(units) - 1:
                f /= 1024.0
                i += 1
            return f"{f:.1f} {units[i]}" if i > 0 else f"{int(f)} {units[i]}"

        entries = os.listdir(path)
        files = [name for name in entries if os.path.isfile(os.path.join(path, name))]
        files.sort()

        results = []
        for name in files:
            full_path = os.path.abspath(os.path.join(path, name))
            try:
                size_bytes = os.path.getsize(full_path)
            except Exception:
                size_bytes = 0
            results.append(
                {
                    "path": full_path,
                    "size": human_size(size_bytes),
                }
            )

        return wrap_dict_to_toon(results)
    except Exception as e:
        return f"Error listing files in '{path}': {e}"


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
            "description": "Read the entire file content. Truncate to max_char (default 100000) if longer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "max_char": {
                        "type": "integer",
                        "description": "Maximum number of characters to return",
                        "default": 100000,
                    },
                },
                "required": ["path"],
            },
        },
    },
)
def read_file(graph_state: GraphState, path: str, max_char: int = 100000) -> str:
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
