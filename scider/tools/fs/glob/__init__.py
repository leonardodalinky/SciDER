"""GlobTool — find files by name pattern."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext

MAX_RESULTS = 200


class GlobInput(BaseModel):
    pattern: str = Field(
        description="The glob pattern to match files against (e.g., '**/*.py', 'src/**/*.ts')"
    )
    path: str | None = Field(
        default=None,
        description="Directory to search in. Defaults to current working directory.",
    )


class GlobTool(BaseTool):
    name = "Glob"
    description = (
        "Fast file pattern matching. Use this to find files by name patterns. "
        "Returns matching file paths sorted by modification time. "
        f"Results are limited to {MAX_RESULTS} files."
    )
    input_schema = GlobInput
    _always_read_only = True
    prompt = (
        "# Glob tool usage\n"
        "- Use Glob to find files by name pattern instead of Bash with `find` or `ls`.\n"
        "- Supports patterns like `**/*.py`, `src/**/*.ts`, `*.json`.\n"
        "- Results are sorted by modification time (most recent first).\n"
    )

    def call(self, context: ToolContext, *, pattern: str, path: str | None = None) -> str:
        search_dir = path or "."
        search_dir = os.path.expandvars(os.path.expanduser(search_dir))

        if not os.path.isdir(search_dir):
            return f"Error: Directory '{search_dir}' does not exist."

        try:
            base = Path(search_dir)
            matches = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

            truncated = len(matches) > MAX_RESULTS
            matches = matches[:MAX_RESULTS]

            if not matches:
                return f"No files found matching pattern '{pattern}' in {search_dir}"

            lines = [str(m) for m in matches]
            if truncated:
                lines.append(
                    f"\n(Results truncated. {MAX_RESULTS} of {len(matches)}+ matches shown. "
                    "Use a more specific pattern.)"
                )

            return "\n".join(lines)
        except Exception as e:
            return f"Error searching for pattern '{pattern}': {e}"
