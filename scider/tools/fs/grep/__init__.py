"""GrepTool — search file contents by regex pattern."""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext

MAX_RESULTS = 200
MAX_CONTEXT_LINES = 3


class GrepInput(BaseModel):
    pattern: str = Field(description="Regular expression pattern to search for")
    path: str | None = Field(
        default=None,
        description="File or directory to search in. Defaults to current working directory.",
    )
    glob: str | None = Field(
        default=None,
        description="Glob pattern to filter files (e.g., '*.py', '**/*.ts')",
    )


class GrepTool(BaseTool):
    name = "Grep"
    description = (
        "Search file contents using regex patterns. "
        "Returns matching lines with file paths and line numbers. "
        "Use the glob parameter to filter which files to search."
    )
    input_schema = GrepInput
    _always_read_only = True

    def call(
        self,
        context: ToolContext,
        *,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> str:
        search_path = path or "."
        search_path = os.path.expandvars(os.path.expanduser(search_path))

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        results = []
        files_searched = 0

        try:
            if os.path.isfile(search_path):
                files_to_search = [Path(search_path)]
            elif os.path.isdir(search_path):
                base = Path(search_path)
                if glob:
                    files_to_search = sorted(base.rglob(glob))
                else:
                    files_to_search = sorted(base.rglob("*"))
                files_to_search = [f for f in files_to_search if f.is_file()]
            else:
                return f"Error: Path '{search_path}' does not exist."

            for file_path in files_to_search:
                if len(results) >= MAX_RESULTS:
                    break

                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    files_searched += 1
                except (OSError, UnicodeDecodeError):
                    continue

                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        results.append(f"{file_path}:{line_num}: {line.rstrip()}")
                        if len(results) >= MAX_RESULTS:
                            break

        except Exception as e:
            return f"Error searching: {e}"

        if not results:
            return f"No matches found for pattern '{pattern}' ({files_searched} files searched)"

        header = f"Found {len(results)} matches ({files_searched} files searched)"
        if len(results) >= MAX_RESULTS:
            header += f" (truncated to {MAX_RESULTS})"

        return header + "\n\n" + "\n".join(results)
